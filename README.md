# RMVGCL - Robust Multi-View Graph Contrastive Learning

RMVGCL: Principled Multi-View and Hierarchical Graph Contrastive Learning for Robust and Effective Representation

## Project Structure

```
├── rmvgcl.py          # Core model implementation
├── attack.py          # Adversarial attack experiments
├── requirements.txt   # Dependencies
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

## Core Code

### `rmvgcl.py`

- **Model (`RMVGCL`)**: three GNN encoders (two local views + one global diffusion view) and one shared MLP to get node embeddings.
- **Classifier (`NodeClassifier`)**: an MLP that takes the fused embedding and outputs node labels.
- **Contrastive head (`Discriminator`)**: a bilinear scorer that distinguishes positive node pairs (same node across views) from negatives (shuffled).
- **Subgraph construction**:
  - `SignificantSubgraphGenerator`: builds a structural subgraph with important nodes/edges.
  - `Semantic subgraph`: builds a semantic subgraph using feature and label similarity.
  - `get_diffusion_adj`: builds a global diffusion graph by heat diffusion.
- **Training (`train_rmvgcl`)**: optimizes contrastive loss + classification loss with early stopping and reports accuracies.

### `attack.py`

- **`poison_data`**: adds noise to node features and perturbs edges.
- **`flip_labels`**: randomly flips a small portion of node labels.
- **`prune_model`**: randomly prunes model weights.
- **`run_attack_experiments`**: runs RMVGCL under different attack combinations and saves results/plots.

### `requirements.txt`

Python dependencies for running RMVGCL and attack experiments.


### Run Full Experiments

```bash
# Run multiple times and average results (default: 10 runs)
python rmvgcl.py
```

### Adversarial Attack Experiments

```bash
python attack.py
```


## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hid_ft` | 256 | Hidden layer dimension |
| `out_ft` | 64 | Output embedding dimension |
| `num_layers` | 2 | Number of GNN layers |
| `dropout` | 0.3 | Dropout rate |
| `lr` | 0.0005 | Learning rate |
| `lambda_cls` | 0.5 | Classification loss weight |
| `patience` | 50 | Early stopping patience |

## Datasets

Supports the following datasets:
- Cora (default)
- CiteSeer
- PubMed
- Amazon Computers

## Baseline


We compare our method with the following representative GNN and graph contrastive learning methods:

- **GAT** – Graph Attention Network.
- **APPNP** – Predict then propagate：Graph neural networks meet personalized pagerank.
- **GPRGNN** – Adaptive universal generalized pagerank graph neural network.
- **DGI** – Deep Graph Infomax.
- **MVGRL** – Multi-View Graph Representation Learning.
- **ReGCL** – Rethinking Message Passing in Graph Contrastive Learning.
