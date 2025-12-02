# PH-Map

PH-Map: Multi-task cell type classification package for single-cell RNA sequencing data.

## Installation

### Quick Install (Recommended)

```bash
# 1. Activate phmap-env environment
mamba activate phmap-env

# 2. Navigate to phmap directory
cd /home/data/fhz/project/phmap_package/phmap

# 3. Install in development mode (recommended)
pip install -e .
```

**Note:** All dependencies are automatically installed via `setup.py`'s `install_requires`. See [INSTALLATION.md](INSTALLATION.md) for detailed instructions.

### Dependencies

The package automatically installs the following dependencies when you run `pip install`:
- `torch>=2.0.0` - PyTorch deep learning framework
- `scanpy>=1.9.0` - Single-cell analysis
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning utilities
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical plotting
- `anndata>=0.9.0` - AnnData data structure
- `scipy>=1.9.0` - Scientific computing

**Python version:** Requires Python >= 3.11

For detailed installation instructions and troubleshooting, see [INSTALLATION.md](INSTALLATION.md).

## Quick Start

### Using Default Model (Recommended)

```python
import phmap
import scanpy as sc

# Load query data
adata = sc.read_h5ad('query_data.h5ad')

# Predict using default model (automatically loads full_model)
result = phmap.predict(adata, return_probabilities=True)

# Visualize prediction probabilities
phmap.pl.plot_probability_bar(result, label_columns=['anno_lv4'])

# Add predictions to AnnData
adata = result.to_adata(adata)
```

### Using Custom Model

```python
import phmap

# Load built-in pretrained model
model = phmap.load_pretrained_model('full_model')

# Or load from a file path
model = phmap.load_pretrained_model('path/to/my_model.pth')

# Predict
result = phmap.predict(model, adata, return_probabilities=True)

# Visualize
phmap.pl.plot_probability_bar(result, label_columns=['anno_lv4'])
```

### Training Your Own Model

```python
import phmap
import scanpy as sc

# Load reference data
adata_train = sc.read_h5ad('reference_data.h5ad')

# Train model
model = phmap.build_model(
    adata=adata_train,
    label_columns=['anno_lv1', 'anno_lv2', 'anno_lv3', 'anno_lv4'], # columns stored in adata.obs
    task_weights=[0.3, 0.8, 1.5, 2.0] # task weights for each categories of label_columns
)

# Save model
model.save('models/my_model.pth')

# Predict on query data
adata_query = sc.read_h5ad("path/to/h5ad/file")
result = phmap.predict(model, adata_query, return_probabilities=True)
```

### Optimized color palettes and Sankey visualization

```python
import phmap
import matplotlib.pyplot as plt

# Assume predictions have been added to adata.obs
# e.g. columns like 'anno_lv4_ref' and 'anno_lv4_pred'

# Generate consistent categorical color palettes for multiple obs columns
palette = phmap.pl.optim_palette(
    adata=adata,
    obs_columns=['anno_lv4_ref', 'anno_lv4_pred'],  # categorical obs columns
    color_scheme="Paired",  # default, similar to RColorBrewer::Paired
)

# Example: visualize mapping between reference and predicted labels with Sankey
x = adata.obs['anno_lv4_ref']
y = adata.obs['anno_lv4_pred']

fig = phmap.pl.sankey(
    x=x,
    y=y,
    colorside="right",  # or "left"
    title="Cell type mapping (anno_lv4)"
)
plt.show()
```

## Available Models

```python
# List available pretrained models
import phmap
models = phmap.list_available_models()
print(models)
```

## Documentation

See `notebooks/` directory for examples:
- [Quick Start](notebooks/quick_start.ipynb) - Quick start of reference mapping using pretrain model on test data

## Features

- **Multi-task Learning**: Simultaneously predict multiple hierarchical cell type levels
- **Easy-to-use API**: Similar to celltypist's interface
- **Automatic Gene Alignment**: Handles gene name matching automatically
- **Pre-trained Models**: Includes full_model trained on complete dataset
- **Rich Visualizations**: Multiple plotting functions for result analysis
- **Optimized Palettes & Sankey**: `phmap.pl.optim_palette` for categorical color schemes and `phmap.pl.sankey` for flow visualization between annotations
- **GPU Support**: Automatic CUDA detection and usage

## Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- scanpy >= 1.9.0
- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## License

See LICENSE file for details.

