import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from models import MVGRL
from data_processing import load_data
from train import train_mvgrl


def poison_data(data, noise_level=0.05, edge_perturbation_ratio=0.01):
 
    x = data['x'].clone()
    noise = torch.randn_like(x) * noise_level
    x += noise

    edge_index = data['edge_index_local1'].clone()
    num_edges = edge_index.size(1)
    num_perturb = int(num_edges * edge_perturbation_ratio)

    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[torch.randperm(num_edges)[:num_perturb]] = False
    edge_index = edge_index[:, mask]

    num_nodes = x.size(0)
    new_edges = torch.randint(0, num_nodes, (2, num_perturb))
    edge_index = torch.cat([edge_index, new_edges], dim=1)

    data['x'] = x
    data['edge_index_local1'] = edge_index
    return data

def prune_model(model, prune_ratio=0.1):

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            mask = torch.rand_like(param) > prune_ratio
            param.data *= mask.float()
    return model

def flip_labels(data, flip_ratio=0.05):

    y = data['y'].clone()
    num_nodes = y.size(0)
    num_flip = int(num_nodes * flip_ratio)

    flip_idx = torch.randperm(num_nodes)[:num_flip]
    for idx in flip_idx:
        y[idx] = torch.randint(0, data['num_classes'], (1,)).item()

    data['y'] = y
    return data

def get_attack_settings():

    return [
        {"name": "baseline", "poison": False, "flip": False, "prune": False},
        {"name": "flip", "poison": False, "flip": True, "prune": False},
        {"name": "prune", "poison": False, "flip": False, "prune": True},
        {"name": "poison_flip", "poison": True, "flip": True, "prune": False},
        {"name": "poison_prune", "poison": True, "flip": False, "prune": True},
        {"name": "flip_prune", "poison": False, "flip": True, "prune": True},
        {"name": "poison_flip_prune", "poison": True, "flip": True, "prune": True},
    ]

def run_attack_experiments(n_runs=10, verbose=False):
    """
    Run adversarial robustness experiments with different attack combinations.
    
    Args:
        n_runs: Number of experimental runs per attack setting
        verbose: Whether to print progress information
    
    Returns:
        Dictionary containing results for all attack settings
    """
    base_output_dir = "attack"
    all_histories = {}
    all_results = {}
    
    for setting in get_attack_settings():
        attack_name = setting["name"]
        if verbose:
            print(f"\n===== Attack: {attack_name} =====")
        
        output_dir = os.path.join(base_output_dir, attack_name)
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for run in range(n_runs):
            data = load_data(dataset_name='cora', diffusion_threshold=1e-3, use_cpu=False)
            
            if setting["poison"]:
                data = poison_data(data, noise_level=0.05, edge_perturbation_ratio=0.01)
            if setting["flip"]:
                data = flip_labels(data, flip_ratio=0.05)
            
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
            
            if setting["prune"]:
                model = prune_model(model, prune_ratio=0.1)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
            
            model, history, metrics = train_mvgrl(
                model=model,
                data=data,
                optimizer=optimizer,
                n_epochs=500,
                eval_every=10,
                patience=50,
                lambda_cls=0.5,
                verbose=False
            )
            
            results.append({
                'val_acc': metrics['best_val_acc'],
                'test_acc': metrics['best_test_acc']
            })
            
            all_histories[attack_name] = history
        
        val_acc_mean = np.mean([r['val_acc'] for r in results])
        val_acc_std = np.std([r['val_acc'] for r in results])
        test_acc_mean = np.mean([r['test_acc'] for r in results])
        test_acc_std = np.std([r['test_acc'] for r in results])
        
        summary = {
            'attack_name': attack_name,
            'val_acc_mean': val_acc_mean,
            'val_acc_std': val_acc_std,
            'test_acc_mean': test_acc_mean,
            'test_acc_std': test_acc_std
        }
        
        all_results[attack_name] = summary
        
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_dir, "summary_results.csv")
        summary_df.to_csv(summary_path, index=False)
        
        if verbose:
            print(f"{attack_name}: Val={val_acc_mean:.4f}±{val_acc_std:.4f}, Test={test_acc_mean:.4f}±{test_acc_std:.4f}")
    
    plot_compare_histories(all_histories, base_output_dir, verbose=verbose)
    
    all_results_df = pd.DataFrame(list(all_results.values()))
    all_results_path = os.path.join(base_output_dir, "all_attacks_summary.csv")
    all_results_df.to_csv(all_results_path, index=False)
    
    return all_results

def plot_compare_histories(histories_dict, base_output_dir, verbose=False):
    """
    Plot and compare training histories across different attack settings.
    
    Args:
        histories_dict: Dictionary of training histories for each attack
        base_output_dir: Base directory to save the comparison plot
        verbose: Whether to print save path information
    
    Returns:
        Path to the saved comparison plot
    """
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for name, hist in histories_dict.items():
        plt.plot(hist['loss'], label=f'{name}')
    plt.title('Loss Comparison Under Different Attacks', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    for name, hist in histories_dict.items():
        epochs = np.arange(0, len(hist['test_acc'])) * 10
        plt.plot(epochs, hist['test_acc'], label=f'{name}')
    plt.title('Test Accuracy Comparison Under Different Attacks', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    compare_path = os.path.join(base_output_dir, 'compare_attack_histories.png')
    plt.savefig(compare_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Comparison plot saved to: {compare_path}")
    
    return compare_path

if __name__ == "__main__":
    run_attack_experiments(n_runs=10, verbose=True) 
