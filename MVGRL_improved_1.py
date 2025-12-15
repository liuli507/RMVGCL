
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

class GNN(nn.Module):

    def __init__(self, in_ft, out_ft, bias=True, act=F.relu):
        super(GNN, self).__init__()
        self.conv = GCNConv(in_ft, out_ft, bias=bias)
        self.act = act

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        return self.act(x) if self.act is not None else x

class Encoder(nn.Module):

    def __init__(self, in_ft, hid_ft, out_ft, num_layers=2, dropout=0.3, act=F.relu):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList([GNN(in_ft, hid_ft, act=None)]) 
        
        for _ in range(num_layers - 2):
            self.layers.append(GNN(hid_ft, hid_ft, act=None))
        
        self.layers.append(GNN(hid_ft, out_ft, act=None))
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hid_ft) for _ in range(num_layers - 1)
        ])
        self.batch_norms.append(nn.BatchNorm1d(out_ft))
        
        self.act = act
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, (layer, norm) in enumerate(zip(self.layers, self.batch_norms)):
            if i != 0:
                x = F.dropout(x, self.dropout, training=self.training)
            x = layer(x, edge_index, edge_weight)
            x = norm(x)  
            x = self.act(x) if self.act is not None else x  
        return x

class Discriminator(nn.Module):

    def __init__(self, ft_size):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(ft_size, ft_size, 1)
        
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h1, h2, h3=None, h4=None, training=True):
        if training:
            sc_pos = self.f_k(h1, h2)  
            h2_neg = h2[torch.randperm(h2.size(0))]
            sc_neg = self.f_k(h1, h2_neg)
            
            logits = torch.cat((sc_pos, sc_neg), 1)
            return logits
        else:
            return self.f_k(h1, h2)


class NodeClassifier(nn.Module):

    def __init__(self, in_ft, num_classes):
        super(NodeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_ft, in_ft // 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_ft // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class SharedMLP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class MVGRL(nn.Module):

    def __init__(self, in_ft, hid_ft, out_ft, num_classes, num_layers=2, act=F.relu, dropout=0.3):
        super(MVGRL, self).__init__()
        self.local_gnn1 = Encoder(in_ft, hid_ft, out_ft, num_layers, dropout, act)  
        self.local_gnn2 = Encoder(in_ft, hid_ft, out_ft, num_layers, dropout, act)  
        self.global_gnn = Encoder(in_ft, hid_ft, out_ft, num_layers, dropout, act)  
        self.shared_mlp = SharedMLP(out_ft, out_ft)
        self.node_graph_discriminator = Discriminator(out_ft)
        self.node_classifier = NodeClassifier(out_ft, num_classes)
        
        self.l2_reg = 1e-4

    def forward(self, x, edge_index_local1, edge_index_local2, edge_index_global, 
                edge_weight_local1=None, edge_weight_local2=None, edge_weight_global=None):
        h_local1 = self.local_gnn1(x, edge_index_local1, edge_weight_local1)  
        h_local2 = self.local_gnn2(x, edge_index_local2, edge_weight_local2)  
        h_global = self.global_gnn(x, edge_index_global, edge_weight_global)  
        
        z_local1 = self.shared_mlp(h_local1)
        z_local2 = self.shared_mlp(h_local2)
        z_global = self.shared_mlp(h_global)
        
        node_pred = self.node_classifier(z_local1 + z_local2 + z_global)
        
        return z_local1, z_local2, z_global, node_pred

    def get_embedding(self, x, edge_index_local1, edge_index_local2, edge_index_global, 
                     edge_weight_local1=None, edge_weight_local2=None, edge_weight_global=None):
        h_local1 = self.local_gnn1(x, edge_index_local1, edge_weight_local1)
        h_local2 = self.local_gnn2(x, edge_index_local2, edge_weight_local2)
        h_global = self.global_gnn(x, edge_index_global, edge_weight_global)
        
        z_local1 = self.shared_mlp(h_local1)
        z_local2 = self.shared_mlp(h_local2)
        z_global = self.shared_mlp(h_global)
        
        return z_local1 + z_local2 + z_global

class SignificantSubgraphGenerator:

    def __init__(self):
        pass
    
    def compute_node_importance(self, x, edge_index, num_nodes):
        adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  
        deg = adj.sum(dim=1)
        deg_inv = torch.pow(deg, -1)
        deg_inv[deg_inv == float('inf')] = 0
        P = adj * deg_inv.unsqueeze(1)
        scores = torch.ones(num_nodes, device=x.device) / num_nodes
        for _ in range(10):  
            scores = 0.85 * torch.mm(P, scores.unsqueeze(1)).squeeze() + 0.15 / num_nodes
        return scores
            
    def compute_edge_importance(self, x, edge_index, node_scores):
        edge_scores = torch.zeros(edge_index.size(1), device=x.device)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            x = F.normalize(x, p=2, dim=1)
            sim = F.cosine_similarity(x[src].unsqueeze(0), x[dst].unsqueeze(0))
            edge_scores[i] = sim * (node_scores[src] + node_scores[dst]) / 2
        return edge_scores
    
    def extract_significant_subgraph(self, x, edge_index, num_nodes, top_k_ratio=0.3,top_m_ratio=0.5):

        node_scores = self.compute_node_importance(x, edge_index, num_nodes)
        
        k = int(num_nodes * top_k_ratio)
        _, top_nodes = torch.topk(node_scores, k)
        
        edge_scores = self.compute_edge_importance(x, edge_index, node_scores)

        mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=x.device)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src in top_nodes and dst in top_nodes:
                mask[i] = True

        filtered_edge_scores = edge_scores[mask]
        filtered_edge_index = edge_index[:, mask]

        m = int(filtered_edge_index.size(1) * top_m_ratio)
        _, top_edges = torch.topk(filtered_edge_scores, m)
        sig_edge_index = filtered_edge_index[:, top_edges]

        return sig_edge_index
    
def augment_graph(edge_index, x, num_nodes, p=0.1):
    sig_generator = SignificantSubgraphGenerator()
    
    sig_edge_index = sig_generator.extract_significant_subgraph(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
        top_k_ratio=p,
        top_m_ratio=0.5
    )
    
    return sig_edge_index

def extract_knowledge_subgraph(edge_index, x, y, num_nodes, k=2, relation_threshold=0.5):

    x_norm = F.normalize(x, p=2, dim=1)
    feature_sim = torch.mm(x_norm, x_norm.t())
    
    label_sim = torch.zeros((num_nodes, num_nodes), device=x.device)
    for i in range(num_nodes):
        label_sim[i] = (y == y[i]).float()
    
    adj = torch.zeros((num_nodes, num_nodes), device=x.device)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    common_neighbors = torch.mm(adj, adj.t())
    common_neighbors = common_neighbors / (common_neighbors.sum(dim=1, keepdim=True) + 1e-8)
    
    semantic_sim = 0.4 * feature_sim + 0.4 * label_sim + 0.2 * common_neighbors
    
    semantic_adj = torch.zeros_like(adj)
    
    for i in range(num_nodes):
        direct_neighbors = torch.where(adj[i] > 0)[0]
        if len(direct_neighbors) == 0:
            continue
        sim_scores = semantic_sim[i, direct_neighbors]
        mask = sim_scores > relation_threshold
        filtered_neighbors = direct_neighbors[mask]
        filtered_scores = sim_scores[mask]
        if filtered_scores.numel() > 0:
            topk = min(5, filtered_scores.numel())
            top_scores, top_indices = torch.topk(filtered_scores, topk)
            top_neighbors = filtered_neighbors[top_indices]
            semantic_adj[i, top_neighbors] = 1
    
    semantic_adj = torch.max(semantic_adj, semantic_adj.t())
    
    semantic_edges = torch.nonzero(semantic_adj).t()
    
    return semantic_edges

def compute_heat(adj, t=5, self_loop=True):
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    
    rowsum = np.array(adj.sum(1)).flatten()  
    d_inv = np.power(rowsum, -1.0)          
    d_inv[np.isinf(d_inv)] = 0.             
    d_inv_mat = sp.diags(d_inv)             
    
    normalized_adj = adj.dot(d_inv_mat)     
    identity = sp.eye(adj.shape[0])
    heat_matrix = identity
    power = identity
    for i in range(1, 10):  
        power = power.dot(normalized_adj) / i
        heat_matrix += (t ** i) * power
    
    return heat_matrix

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def get_diffusion_adj(edge_index, num_nodes, alpha=0.15, threshold=1e-4):

    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), 
                         (edge_index[0].numpy(), edge_index[1].numpy())),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)
    
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    diff_adj = compute_heat(adj, t=alpha)
    
    diff_adj.data[diff_adj.data < threshold] = 0
    diff_adj.eliminate_zeros()
    
    diff_adj_coo = diff_adj.tocoo()
    diff_edge_index = torch.tensor(np.vstack([diff_adj_coo.row, diff_adj_coo.col]), dtype=torch.long)
    diff_edge_weight = torch.tensor(diff_adj_coo.data, dtype=torch.float)
    
    return diff_edge_index, diff_edge_weight

def train_mvgrl(model, data, optimizer, n_epochs=200, eval_every=10, patience=50, lambda_cls=0.5, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    x = data['x'].to(device)
    edge_index_local1 = data['edge_index_local1'].to(device)
    edge_index_local2 = data['edge_index_local2'].to(device)
    edge_index_global = data['edge_index_global'].to(device)
    edge_weight_global = data['edge_weight_global'].to(device) if 'edge_weight_global' in data else None
    y = data['y'].to(device)
    
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        num_nodes = x.shape[0]
        train_idx, temp_idx = train_test_split(np.arange(num_nodes), test_size=0.4, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    history = {
        'loss': [],
        'cls_loss': [],
        'contrast_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }
    
    best_val_acc = 0
    best_epoch = 0
    best_metrics = None
    
    iterator = tqdm(range(n_epochs)) if verbose else range(n_epochs)
    for epoch in iterator:
        model.train()
        optimizer.zero_grad()
        
        edge_index_aug = augment_graph(edge_index_local1.cpu(), x.cpu(), num_nodes).to(device)
        
        h_local1, h_local2, h_global, node_pred = model(
            x, edge_index_aug, edge_index_local2, edge_index_global, None, None, edge_weight_global
        )
        
        local_local_logits = model.node_graph_discriminator(h_local1, h_local2)
        local_local_loss = F.cross_entropy(local_local_logits, torch.zeros(x.shape[0], dtype=torch.long, device=device))
        
        h_local_fused = (h_local1 + h_local2) / 2
        
        local_global_logits = model.node_graph_discriminator(h_local_fused, h_global)
        local_global_loss = F.cross_entropy(local_global_logits, torch.zeros(x.shape[0], dtype=torch.long, device=device))
        
        local_weight = 0.5  
        global_weight = 0.5  
        
        contrast_loss = local_weight * local_local_loss + global_weight * local_global_loss
        
        cls_loss = F.cross_entropy(node_pred[train_mask], y[train_mask])
        
        l2_reg_loss = 0.0
        for param in model.parameters():
            l2_reg_loss += torch.norm(param, 2)
        l2_reg_loss = model.l2_reg * l2_reg_loss
        
        loss = contrast_loss + lambda_cls * cls_loss + l2_reg_loss
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        history['loss'].append(loss.item())
        history['cls_loss'].append(cls_loss.item())
        history['contrast_loss'].append(contrast_loss.item())
        
        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                embedding = model.get_embedding(
                    x, edge_index_local1, edge_index_local2, edge_index_global, 
                    None, None, edge_weight_global
                ).detach().cpu().numpy()
                
                train_acc, val_acc, test_acc = evaluate_embedding_final(
                    embedding, y.cpu().numpy(), train_idx, val_idx, test_idx
                )
                
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['test_acc'].append(test_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_metrics = {
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'test_acc': test_acc,
                        'epoch': epoch + 1
                    }
                    best_model_state = {key: value.cpu() for key, value in model.state_dict().items()}
                elif epoch - best_epoch >= patience:
                    model.load_state_dict(best_model_state)
                    model = model.to(device)
                    break
    
    model.eval()
    with torch.no_grad():
        embedding = model.get_embedding(
            x, edge_index_local1, edge_index_local2, edge_index_global, 
            None, None, edge_weight_global
        ).detach().cpu().numpy()
        final_train_acc, final_val_acc, final_test_acc = evaluate_embedding_final(
            embedding, y.cpu().numpy(), train_idx, val_idx, test_idx
        )
    
    final_metrics = {
        'final_train_acc': best_metrics['train_acc'],
        'final_val_acc': best_metrics['val_acc'],
        'final_test_acc': best_metrics['test_acc'],
        'best_epoch': best_metrics['epoch'],
        'best_val_acc': best_metrics['val_acc'],
        'best_test_acc': best_metrics['test_acc'],
        'final_loss': history['loss'][-1],
        'final_cls_loss': history['cls_loss'][-1],
        'final_contrast_loss': history['contrast_loss'][-1],
        'lambda_cls': lambda_cls,
        'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return model, history, final_metrics

def evaluate_embedding_final(embedding, labels, train_idx, val_idx, test_idx):
    clf = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=500)
    clf.fit(embedding[train_idx], labels[train_idx])
    
    train_acc = clf.score(embedding[train_idx], labels[train_idx])
    val_acc = clf.score(embedding[val_idx], labels[val_idx])
    test_acc = clf.score(embedding[test_idx], labels[test_idx])
    
    return train_acc, val_acc, test_acc

def load_data(dataset_name='cora', diffusion_threshold=1e-4, use_cpu=False):
    dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name)
    data = dataset[0]
    x = data.x
    edge_index = data.edge_index
    y = data.y
    
    if dataset_name == 'citeseer':
        x = preprocess_features(x.numpy())
        x = torch.FloatTensor(x)
    
    idx_train = np.argwhere(data.train_mask.cpu().numpy() == 1).reshape(-1)
    idx_val = np.argwhere(data.val_mask.cpu().numpy() == 1).reshape(-1)
    idx_test = np.argwhere(data.test_mask.cpu().numpy() == 1).reshape(-1)
    
    sig_generator = SignificantSubgraphGenerator()
    
    edge_index_local1 = sig_generator.extract_significant_subgraph(
        x=x,
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        top_k_ratio=0.3,  
        top_m_ratio=0.5   
    )
    edge_index_local2 = extract_knowledge_subgraph(edge_index, x, y, data.num_nodes, k=1)
    
    edge_index_ppr, edge_weight_ppr = get_diffusion_adj(
        edge_index, data.num_nodes, alpha=0.15, threshold=diffusion_threshold
    )
   
    if use_cpu:
        x = x.cpu()
        edge_index_local1 = edge_index_local1.cpu()
        edge_index_local2 = edge_index_local2.cpu()
        edge_index_ppr = edge_index_ppr.cpu()
        edge_weight_ppr = edge_weight_ppr.cpu()
        y = y.cpu()
    
    return {
        'x': x,
        'edge_index_local1': edge_index_local1,
        'edge_index_local2': edge_index_local2,
        'edge_index_global': edge_index_ppr,
        'edge_weight_global': edge_weight_ppr,
        'y': y,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
        'num_features': data.num_features,
        'num_classes': dataset.num_classes
    }

def save_results_to_csv(history, final_metrics, output_dir):
    train_history_df = pd.DataFrame({
        'epoch': range(1, len(history['loss']) + 1),
        'loss': history['loss'],
        'cls_loss': history['cls_loss'],
        'contrast_loss': history['contrast_loss'],
        'val_acc': [None] * (len(history['loss']) - len(history['val_acc'])) + history['val_acc'],
        'test_acc': [None] * (len(history['loss']) - len(history['test_acc'])) + history['test_acc']
    })
    
    history_path = os.path.join(output_dir, f"training_history.csv")
    train_history_df.to_csv(history_path, index=False)
    
    final_metrics_df = pd.DataFrame([final_metrics])
    metrics_path = os.path.join(output_dir, "final_metrics.csv")
    
    if os.path.exists(metrics_path):
        final_metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
    else:
        final_metrics_df.to_csv(metrics_path, index=False)

def main(verbose=False, num_runs=10):
    use_cpu = False  
    device = torch.device('cpu' if use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    output_dir = "ss_11/cora/output/mvgrl_improved_8"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for run in range(num_runs):
        data = load_data(dataset_name='cora', diffusion_threshold=1e-3, use_cpu=use_cpu)
        in_channels = data['num_features']
        hidden_channels = 256  
        out_channels = 64  
        num_classes = data['num_classes']
        
        model = MVGRL(
            in_ft=in_channels,
            hid_ft=hidden_channels,
            out_ft=out_channels,
            num_classes=num_classes,
            num_layers=2,
            dropout=0.3
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4) 
        
        model, history, metrics = train_mvgrl(
            model=model,
            data=data,
            optimizer=optimizer,
            n_epochs=200,
            eval_every=10,
            patience=50,
            lambda_cls=0.5,
            verbose=verbose
        )
        
        results.append({
            'val_acc': metrics['best_val_acc'],
            'test_acc': metrics['best_test_acc']
        })
        
        if verbose:
            print(f"\nRun {run + 1}/{num_runs} - Val: {metrics['best_val_acc']:.4f}, Test: {metrics['best_test_acc']:.4f}")
    
    val_acc_mean = np.mean([r['val_acc'] for r in results])
    val_acc_std = np.std([r['val_acc'] for r in results])
    test_acc_mean = np.mean([r['test_acc'] for r in results])
    test_acc_std = np.std([r['test_acc'] for r in results])
    
    summary = {
        'val_acc_mean': val_acc_mean,
        'val_acc_std': val_acc_std,
        'test_acc_mean': test_acc_mean,
        'test_acc_std': test_acc_std
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    
    if verbose:
        print(f"\nFinal Results: Val Acc = {val_acc_mean:.4f} ± {val_acc_std:.4f}, "
              f"Test Acc = {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    
    return summary

if __name__ == "__main__":
    main(verbose=True)
