import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, dense_to_sparse, add_self_loops, degree
import numpy as np
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# 设置随机种子确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

class GCN(nn.Module):
    """
    图卷积网络编码器
    """
    def __init__(self, in_ft, out_ft, bias=True, act=F.relu):
        super(GCN, self).__init__()
        self.conv = GCNConv(in_ft, out_ft, add_self_loops=False)
        self.act = act

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        return self.act(x) if self.act is not None else x

class Encoder(nn.Module):
    """
    编码器 - 多层GCN，添加正则化
    """
    def __init__(self, in_ft, hid_ft, out_ft, num_layers=2, dropout=0.3, act=F.relu):  # 增加dropout率
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 输入层
        self.layers = nn.ModuleList([GCN(in_ft, hid_ft, act=act)])
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GCN(hid_ft, hid_ft, act=act))
        
        # 输出层
        self.layers.append(GCN(hid_ft, out_ft, act=None))
        
        # 添加BatchNorm层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hid_ft) for _ in range(num_layers - 1)
        ])
        self.batch_norms.append(nn.BatchNorm1d(out_ft))
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, (layer, norm) in enumerate(zip(self.layers, self.batch_norms)):
            if i != 0:
                x = F.dropout(x, self.dropout, training=self.training)
            x = layer(x, edge_index, edge_weight)
            x = norm(x)
        return x

class Discriminator(nn.Module):
    """
    判别器 - 基于双线性模型计算互信息
    """
    def __init__(self, ft_size):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(ft_size, ft_size))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, h1, h2, h3=None, h4=None, training=True):
        if training:
            # 正样本
            sc_1 = torch.sum(h2 * torch.mm(h1, self.weight), dim=1)
            # 负样本 - 打乱节点顺序
            sc_2 = torch.sum(h2[torch.randperm(h2.size(0))] * torch.mm(h1, self.weight), dim=1)
            logits = torch.stack([sc_1, sc_2], dim=1)
            return logits
        else:
            # 用于评估 - 返回正样本得分
            sc_1 = torch.sum(h2 * torch.mm(h1, self.weight), dim=1)
            return sc_1

class NodeClassifier(nn.Module):
    """
    节点分类器 - 用于多任务学习
    """
    def __init__(self, in_ft, num_classes):
        super(NodeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_ft, in_ft // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_ft // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class SharedMLP(nn.Module):
    """
    共享MLP，用于节点和图级特征
    """
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
    """
    严格按照论文结构图实现的MVGRL模型
    """
    def __init__(self, in_ft, hid_ft, out_ft, num_classes, num_layers=2, act=F.relu, dropout=0.1):
        super(MVGRL, self).__init__()
        # 两个GNN编码器
        self.local_gnn = Encoder(in_ft, hid_ft, out_ft, num_layers, dropout, act)
        self.global_gnn = Encoder(in_ft, hid_ft, out_ft, num_layers, dropout, act)
        # 共享MLP
        self.shared_mlp = SharedMLP(out_ft, out_ft)
        # 判别器
        self.node_graph_discriminator = Discriminator(out_ft)
        # 节点分类器
        self.node_classifier = NodeClassifier(out_ft, num_classes)

    def global_pool(self, x):
        return torch.sum(x, dim=0, keepdim=True)  # sum pooling

    def forward(self, x, edge_index_local, edge_index_global, edge_weight_local=None, edge_weight_global=None):
        # 两视图分别编码
        h_local = self.local_gnn(x, edge_index_local, edge_weight_local)
        h_global = self.global_gnn(x, edge_index_global, edge_weight_global)
        # 共享MLP
        z_local = self.shared_mlp(h_local)
        z_global = self.shared_mlp(h_global)
        # 图级池化+MLP
        g_local = self.shared_mlp(self.global_pool(z_local))
        g_global = self.shared_mlp(self.global_pool(z_global))
        # 节点分类预测
        node_pred = self.node_classifier(z_local + z_global)
        return z_local, z_global, g_local, g_global, node_pred

    def get_embedding(self, x, edge_index_local, edge_weight_local=None, edge_index_global=None, edge_weight_global=None):
        # 获取节点嵌入
        h_local = self.local_gnn(x, edge_index_local, edge_weight_local)
        h_global = self.global_gnn(x, edge_index_global, edge_weight_global)
        z_local = self.shared_mlp(h_local)
        z_global = self.shared_mlp(h_global)
        return z_local + z_global

def diffusion(adj, mode='ppr', alpha=0.15, threshold=1e-4):
    """
    计算图扩散矩阵，添加阈值以减少内存占用
    
    参数:
        adj: 邻接矩阵
        mode: 'ppr' 个性化PageRank 或 'heat' 热核
        alpha: 重启概率(PPR)或扩散时间(热核)
        threshold: 小于此值的边权重将被剪枝
    
    返回:
        稀疏扩散矩阵
    """
    if sp.issparse(adj):
        # 获取稀疏邻接矩阵
        adj_coo = adj.tocoo()
        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        # 归一化邻接矩阵: D^-1 A
        mx = r_mat_inv.dot(adj)
    else:
        # 获取密集邻接矩阵
        mx = adj / adj.sum(1, keepdims=True)
        mx = np.nan_to_num(mx)
    
    # 计算扩散矩阵
    if mode == "ppr":
        # 个性化 PageRank: (1-alpha)(I-(1-alpha)D^-1A)^-1
        # 使用迭代近似计算扩散矩阵，而不是直接求逆
        diff_mx = alpha * np.eye(mx.shape[0])
        power = mx
        for _ in range(30):  # 30次迭代通常足够收敛
            diff_mx += (1-alpha) * alpha * power
            power = power.dot(mx)
    elif mode == "heat":
        # 热核: exp(-alpha*L) = exp(-alpha*(I-D^-1A))
        # 使用泰勒展开近似指数
        identity = np.eye(mx.shape[0])
        lap = identity - mx
        diff_mx = identity
        power = identity
        for i in range(1, 10):  # 10阶泰勒展开
            power = power.dot(-alpha * lap) / i
            diff_mx += power
    else:
        raise ValueError(f"Unknown diffusion mode: {mode}")
    
    # 稀疏化 - 剪枝小权重
    if not sp.issparse(diff_mx):
        diff_mx[diff_mx < threshold] = 0
        diff_mx = sp.csr_matrix(diff_mx)
    else:
        diff_mx.data[diff_mx.data < threshold] = 0
        diff_mx.eliminate_zeros()
    
    return diff_mx

def approximate_ppr(adj, alpha=0.15, epsilon=1e-4, max_iter=100):
    """
    近似个性化PageRank (PPR) 计算
    
    参数:
        adj: 邻接矩阵
        alpha: 重启概率
        epsilon: 收敛阈值
        max_iter: 最大迭代次数
    
    返回:
        稀疏PPR矩阵
    """
    n = adj.shape[0]
    if sp.issparse(adj):
        # 获取稀疏邻接矩阵
        adj_coo = adj.tocoo()
        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        # 归一化邻接矩阵: D^-1 A
        mx = r_mat_inv.dot(adj)
    else:
        # 获取密集邻接矩阵
        mx = adj / adj.sum(1, keepdims=True)
        mx = np.nan_to_num(mx)
    
    # 初始化PPR矩阵
    ppr = alpha * sp.eye(n)
    residual = sp.eye(n)
    
    # 迭代计算PPR
    for _ in range(max_iter):
        # 更新残差
        residual = (1 - alpha) * residual.dot(mx)
        # 更新PPR
        ppr += alpha * residual
        
        # 检查收敛
        if residual.nnz == 0 or np.max(np.abs(residual.data)) < epsilon:
            break
    
    return ppr

def get_diffusion_adj(edge_index, num_nodes, mode='approximate_ppr', alpha=0.15, epsilon=1e-4, threshold=1e-4):
    """
    从边索引创建扩散邻接矩阵，使用改进的扩散方法
    
    参数:
        edge_index: 边索引 [2, num_edges]
        num_nodes: 节点数量
        mode: 扩散模式 ('approximate_ppr' 或 'heat')
        alpha: 扩散参数
        epsilon: PPR收敛阈值
        threshold: 小于此值的边权重将被剪枝
    
    返回:
        扩散边索引和权重
    """
    # 创建稀疏邻接矩阵
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), 
                         (edge_index[0].numpy(), edge_index[1].numpy())),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)
    
    # 确保对称性 (无向图)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # 添加自环
    adj = adj + sp.eye(adj.shape[0])
    
    # 计算扩散矩阵
    if mode == "approximate_ppr":
        diff_adj = approximate_ppr(adj, alpha=alpha, epsilon=epsilon)
    else:
        diff_adj = diffusion(adj, mode=mode, alpha=alpha, threshold=threshold)
    
    # 稀疏化 - 剪枝小权重
    diff_adj.data[diff_adj.data < threshold] = 0
    diff_adj.eliminate_zeros()
    
    # 转换回边索引格式
    diff_adj_coo = diff_adj.tocoo()
    diff_edge_index = torch.tensor(np.vstack([diff_adj_coo.row, diff_adj_coo.col]), dtype=torch.long)
    diff_edge_weight = torch.tensor(diff_adj_coo.data, dtype=torch.float)
    
    return diff_edge_index, diff_edge_weight

def global_add_pool(x, batch):
    """全局加和池化"""
    if batch is None:
        # 单图情况下直接对所有节点求和
        return torch.sum(x, dim=0, keepdim=True)
    else:
        # 批处理情况
        batch_size = int(batch.max()) + 1
        return torch.stack([torch.sum(x[batch == i], dim=0) for i in range(batch_size)])

def augment_graph(edge_index, num_nodes, p=0.1):
    """
    图数据增强：随机添加和删除边
    
    参数:
        edge_index: 原始边索引
        num_nodes: 节点数量
        p: 边的修改概率
    
    返回:
        增强后的边索引
    """
    # 转换为集合以便快速查找
    edge_set = set(map(tuple, edge_index.t().numpy()))
    
    # 随机删除边
    edges_to_remove = []
    for edge in edge_set:
        if np.random.random() < p:
            edges_to_remove.append(edge)
    edge_set = edge_set - set(edges_to_remove)
    
    # 随机添加边
    for _ in range(int(len(edge_set) * p)):
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j and (i, j) not in edge_set:
            edge_set.add((i, j))
    
    # 转换回边索引格式
    edge_index_aug = torch.tensor(list(edge_set), dtype=torch.long).t()
    return edge_index_aug

def train_mvgrl(model, data, optimizer, n_epochs=200, eval_every=10, patience=50, lambda_cls=0.5):
    """
    训练改进的MVGRL模型，包含多任务学习和数据增强
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    x = data['x'].to(device)
    edge_index_local = data['edge_index_local'].to(device)
    edge_index_global = data['edge_index_global'].to(device)
    edge_weight_global = data['edge_weight_global'].to(device) if 'edge_weight_global' in data else None
    y = data['y'].to(device)
    
    # 训练/验证/测试分割
    num_nodes = x.shape[0]
    train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=0.8, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # 训练历史
    history = {'loss': [], 'cls_loss': [], 'contrast_loss': [], 'val_acc': [], 'test_acc': []}
    
    # 早停机制
    best_val_acc = 0
    best_epoch = 0
    
    # 训练循环
    for epoch in tqdm(range(n_epochs)):
        model.train()
        optimizer.zero_grad()
        
        # 数据增强
        edge_index_aug = augment_graph(edge_index_local.cpu(), num_nodes).to(device)
        
        # 前向传播
        h_local, h_global, g_local, g_global, node_pred = model(x, edge_index_aug, edge_index_global, None, edge_weight_global)
        
        # 节点-图级对比损失
        node_graph_logits = model.node_graph_discriminator(h_local, g_global)
        node_graph_loss = F.cross_entropy(node_graph_logits, torch.zeros(x.shape[0], dtype=torch.long, device=device))
        
        # 节点-节点对比损失
        node_node_logits = model.node_graph_discriminator(h_local, h_global)
        node_node_loss = F.cross_entropy(node_node_logits, torch.zeros(x.shape[0], dtype=torch.long, device=device))
        
        # 对比损失
        contrast_loss = node_graph_loss + node_node_loss
        
        # 节点分类损失
        cls_loss = F.cross_entropy(node_pred[train_mask], y[train_mask])
        
        # 总损失
        loss = contrast_loss + lambda_cls * cls_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        history['loss'].append(loss.item())
        history['cls_loss'].append(cls_loss.item())
        history['contrast_loss'].append(contrast_loss.item())
        
        # 评估
        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                # 获取节点嵌入
                embedding = model.get_embedding(x, edge_index_local, None, edge_index_global, edge_weight_global).detach().cpu().numpy()
                
                # 用嵌入训练线性分类器
                val_acc, test_acc = evaluate_embedding(embedding, y.cpu().numpy(), train_idx, val_idx, test_idx)
                
                history['val_acc'].append(val_acc)
                history['test_acc'].append(test_acc)
                
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Cls Loss = {cls_loss.item():.4f}, "
                      f"Contrast Loss = {contrast_loss.item():.4f}, Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}")
                
                # 早停检查
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    # 保存最佳模型
                    best_model_state = {key: value.cpu() for key, value in model.state_dict().items()}
                elif epoch - best_epoch >= patience:
                    print(f"Early stopping at epoch {epoch+1}, best epoch: {best_epoch+1}")
                    # 恢复最佳模型
                    model.load_state_dict(best_model_state)
                    model = model.to(device)
                    break
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        embedding = model.get_embedding(x, edge_index_local, None, edge_index_global, edge_weight_global).detach().cpu().numpy()
        train_acc, val_acc, test_acc = evaluate_embedding_final(embedding, y.cpu().numpy(), train_idx, val_idx, test_idx)
        print(f"Final: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}")
    
    return model, history

def evaluate_embedding(embedding, labels, train_idx, val_idx, test_idx):
    """使用线性分类器评估嵌入"""
    # 训练线性分类器
    clf = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=500)
    clf.fit(embedding[train_idx], labels[train_idx])
    
    # 评估
    val_acc = clf.score(embedding[val_idx], labels[val_idx])
    test_acc = clf.score(embedding[test_idx], labels[test_idx])
    
    return val_acc, test_acc

def evaluate_embedding_final(embedding, labels, train_idx, val_idx, test_idx):
    """最终评估，返回训练、验证和测试精度"""
    # 训练线性分类器
    clf = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=500)
    clf.fit(embedding[train_idx], labels[train_idx])
    
    # 评估
    train_acc = clf.score(embedding[train_idx], labels[train_idx])
    val_acc = clf.score(embedding[val_idx], labels[val_idx])
    test_acc = clf.score(embedding[test_idx], labels[test_idx])
    
    return train_acc, val_acc, test_acc

def plot_training_history(history, save_path='mvgrl_training_history.png'):
    """绘制训练历史，包含损失函数和测试准确率"""
    plt.figure(figsize=(15, 6))
    
    # 绘制损失函数
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='总损失', color='red')
    plt.plot(history['cls_loss'], label='分类损失', color='blue')
    plt.plot(history['contrast_loss'], label='对比损失', color='green')
    plt.title('模型损失变化', fontsize=14)
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制测试准确率柱形图
    plt.subplot(1, 2, 2)
    epochs = np.arange(0, len(history['test_acc'])) * 10
    test_acc = history['test_acc']
    
    # 找到最佳验证集对应的测试准确率
    best_val_idx = np.argmax(history['val_acc'])
    best_test_acc = test_acc[best_val_idx]
    best_epoch = epochs[best_val_idx]
    
    # 绘制柱状图
    bars = plt.bar(epochs, test_acc, width=8, color='skyblue', alpha=0.7)
    
    # 标注最佳测试准确率
    plt.plot(best_epoch, best_test_acc, 'r*', markersize=15, label=f'最佳测试准确率: {best_test_acc:.4f}')
    
    plt.title('测试准确率随训练轮数的变化', fontsize=14)
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('测试准确率', fontsize=12)
    plt.ylim(min(test_acc) - 0.05, max(test_acc) + 0.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_data(dataset_name='cora', diffusion_threshold=1e-4, use_cpu=False):
    """加载数据集，支持ADJ-PPR双视图"""
    dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name)
    data = dataset[0]
    x = data.x
    edge_index = data.edge_index
    y = data.y
    # 局部视图：原始邻接
    edge_index_local = edge_index
    # 全局视图：PPR扩散
    print("计算PPR扩散矩阵...")
    edge_index_ppr, edge_weight_ppr = get_diffusion_adj(
        edge_index, data.num_nodes, mode='approximate_ppr', alpha=0.15, threshold=diffusion_threshold
    )
    print(f"原始边数: {edge_index.shape[1]}, PPR扩散边数: {edge_index_ppr.shape[1]}")
    if use_cpu:
        x = x.cpu()
        edge_index_local = edge_index_local.cpu()
        edge_index_ppr = edge_index_ppr.cpu()
        edge_weight_ppr = edge_weight_ppr.cpu()
        y = y.cpu()
    return {
        'x': x,
        'edge_index_local': edge_index_local,
        'edge_index_global': edge_index_ppr,
        'edge_weight_global': edge_weight_ppr,
        'y': y,
        'num_features': data.num_features,
        'num_classes': dataset.num_classes
    }

def save_results_to_csv(history, final_metrics, output_dir):
    """
    将训练结果保存为CSV文件，追加新的结果到现有文件
    
    参数:
        history: 训练历史字典
        final_metrics: 最终评估指标字典
        output_dir: 输出目录
    """
    # 创建训练历史DataFrame
    train_history_df = pd.DataFrame({
        'epoch': range(1, len(history['loss']) + 1),
        'loss': history['loss'],
        'cls_loss': history['cls_loss'],
        'contrast_loss': history['contrast_loss'],
        'val_acc': [None] * (len(history['loss']) - len(history['val_acc'])) + history['val_acc'],
        'test_acc': [None] * (len(history['loss']) - len(history['test_acc'])) + history['test_acc']
    })
    
    # 保存训练历史（每次运行生成新的训练历史文件）
    history_path = os.path.join(output_dir, f"training_history.csv")
    train_history_df.to_csv(history_path, index=False)
    print(f"训练历史已保存到: {history_path}")
    
    # 创建最终评估指标DataFrame
    final_metrics_df = pd.DataFrame([final_metrics])
    
    # 最终评估指标文件路径
    metrics_path = os.path.join(output_dir, "final_metrics.csv")
    
    # 检查文件是否存在
    if os.path.exists(metrics_path):
        # 如果文件存在，追加新结果
        final_metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
    else:
        # 如果文件不存在，创建新文件
        final_metrics_df.to_csv(metrics_path, index=False)

def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def run_single_experiment(seed=None, output_dir=None, save_csv=False, verbose=True):
    if seed is not None:
        set_global_seed(seed)
    use_gpu = torch.cuda.is_available()
    use_cpu = False
    if use_cpu:
        device = torch.device('cpu')
        if verbose:
            print("强制使用CPU")
    else:
        device = torch.device('cuda' if use_gpu else 'cpu')
        if verbose:
            print(f"使用设备: {device}")
    if output_dir is None:
        output_dir = "ss_11/cora/output/mvgrl_improved_1"
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"结果将保存到: {output_dir}")
    if verbose:
        print("加载数据...")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    if verbose:
        print("训练模型...")
    try:
        model, history = train_mvgrl(
            model=model,
            data=data,
            optimizer=optimizer,
            n_epochs=200,
            eval_every=10,
            patience=50,
            lambda_cls=0.5
        )
        history_plot_path = os.path.join(output_dir, "training_history.png")
        plot_training_history(history, save_path=history_plot_path)
        if verbose:
            print(f"训练历史图表已保存到: {history_plot_path}")
        final_metrics = {
            'final_train_acc': history['val_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'final_test_acc': history['test_acc'][-1],
            'best_epoch': np.argmax(history['val_acc']) * 10 + 1,
            'best_val_acc': max(history['val_acc']),
            'best_test_acc': history['test_acc'][np.argmax(history['val_acc'])],
            'final_loss': history['loss'][-1],
            'final_cls_loss': history['cls_loss'][-1],
            'final_contrast_loss': history['contrast_loss'][-1],
            'lambda_cls': 0.5,
            'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if save_csv:
            save_results_to_csv(history, final_metrics, output_dir)
        return history['test_acc'][-1]
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n内存不足错误。尝试以下解决方案:")
            print("1. 将 use_cpu 设置为 True")
            print("2. 进一步增加 diffusion_threshold 值 (如 1e-2)")
            print("3. 进一步减小 hidden_channels 和 out_channels")
        else:
            raise e
        return None

def run_experiments(n=10, base_seed=42):
    results = []
    for i in range(n):
        print(f"\n===== 第{i+1}次实验 =====")
        test_acc = run_single_experiment(seed=base_seed + i, output_dir=f"ss_11/cora/output/mvgrl_improved_1/exp_{i+1}", save_csv=False, verbose=False)
        if test_acc is not None:
            print(f"第{i+1}次 test_acc: {test_acc:.4f}")
            results.append(test_acc)
        else:
            print(f"第{i+1}次运行失败")
    if results:
        mean = np.mean(results)
        std = np.std(results)
        print(f"\n10次实验 test_acc 平均值: {mean:.4f}，标准差: {std:.4f}")
    else:
        print("所有实验均失败")

def main():
    run_single_experiment()

if __name__ == "__main__":
    run_experiments(n=10)