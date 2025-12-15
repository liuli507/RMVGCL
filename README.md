# RMVGCL - Robust Multi-View Graph Contrastive Learning

A multi-view graph contrastive learning approach for node classification with adversarial robustness evaluation.

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

## Core Components

### rmvgcl.py

- **GNN / Encoder**: Multi-layer GCN encoder with batch normalization and dropout
- **Discriminator**: Node-graph discriminator for contrastive learning
- **NodeClassifier**: Classification head for node classification
- **SignificantSubgraphGenerator**: Extracts significant subgraphs based on PageRank and feature similarity
- **Knowledge Subgraph Extraction**: Builds semantic subgraphs using feature similarity, label similarity, and common neighbors
- **Heat Diffusion**: Computes global diffusion adjacency matrix

**Key Features**:
- Three-view learning: Significant subgraph view + Knowledge subgraph view + Global diffusion view
- Joint training with contrastive loss and classification loss
- Early stopping mechanism

### attack.py

Adversarial robustness evaluation:

| Attack Method | Description |
|--------------|-------------|
| `poison_data` | Data poisoning: feature noise injection + edge perturbation |
| `flip_labels` | Label flipping: randomly flip node labels |
| `prune_model` | Model pruning: randomly zero out model weights |

**Experiment Settings**:
- baseline (no attack)
- flip (label flipping only)
- prune (model pruning only)
- poison_flip (poisoning + flipping)
- poison_prune (poisoning + pruning)
- flip_prune (flipping + pruning)
- poison_flip_prune (all combined)

## Quick Start

### Train Model

```python
from rmvgcl import load_data, RMVGCL, train_rmvgcl
import torch

# Load data
data = load_data(dataset_name='cora')

# Initialize model
model = RMVGCL(
    in_ft=data['num_features'],
    hid_ft=256,
    out_ft=64,
    num_classes=data['num_classes'],
    num_layers=2,
    dropout=0.3
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
model, history, metrics = train_rmvgcl(
    model=model,
    data=data,
    optimizer=optimizer,
    n_epochs=200,
    verbose=True
)

print(f"Test Accuracy: {metrics['best_test_acc']:.4f}")
```

### Run Full Experiments

```bash
# Run multiple times and average results (default: 10 runs)
python rmvgcl.py
```

### Adversarial Attack Experiments

```bash
python attack.py
```

Results are saved in the `attack/` directory, including:
- Summary results for each attack setting (CSV)
- Training curve comparison plots across different attacks

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

Supports Planetoid datasets:
- Cora (default)
- CiteSeer
- PubMed

## Outputs

After training:
- `rmvgcl/summary_results.csv` - Averaged results over multiple runs
- `attack/all_attacks_summary.csv` - Comparison results across all attack settings
- `attack/compare_attack_histories.png` - Attack comparison visualization
