# PH-Map

PH-Map: Multi-task cell type classification package for single-cell RNA sequencing data.

## Installation

### Quick Install (Recommended)

```bash
# 1. Create and activate phmap-env environment
mamba create -n phmap-env python=3.11 -y
mamba activate phmap-env

# 2. Install the dependency package for phmap
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install torch_geometric -i https://mirrors.nju.edu.cn/pypi/web/simple
uv pip install scanpy==1.11.5 pandas==1.5.3 numpy==1.26.4 -i https://mirrors.nju.edu.cn/pypi/web/simple
uv pip install scikit-learn matplotlib seaborn scipy pytest black flake8 -i https://mirrors.nju.edu.cn/pypi/web/simple

# 3. Navigate to phmap directory
git clone https://github.com/Doctorluka/PH-Map.git
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

## Test data and models

The full machine learning model, trained on the comprehensive single-cell RNA-sequencing (scRNA-seq) dataset for pulmonary hypertension, is available in FigShare (10.6084/m9.figshare.30666551) and Zenodo (10.5281/zenodo.17661644). The test datasets utilized in the accompanying analysis notebooks are also deposited at FigShare (10.6084/m9.figshare.30666551).

## Quick Start

### Using Default Model (Recommended)

```python
import phmap
import scanpy as sc

# Load query data
adata = sc.read_h5ad('query_data.h5ad')

# Predict using default model (automatically loads full_model)
result = phmap.predict(adata, return_probabilities=True)

# Visualize
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

# Predict
adata_query = sc.read_h5ad("path/to/h5ad/file")
result = phmap.predict(model, adata_query, return_probabilities=True)
```

## Available Models

```python
# List available pretrained models
import phmap
models = phmap.list_available_models()
print(models)
```

## Documentation

- [README](docs/README.md) - Overview

See `notebooks/` directory for examples:

- [Quick Start](notebooks/quick_start.ipynb) - Quick start of reference mapping using pretrain model on test data
- [Example for real dataset](notebooks/example_for_real_dataset.ipynb) - Explore GSE228643 datasets

## Features

- **Multi-task Learning**: Simultaneously predict multiple hierarchical cell type levels
- **Easy-to-use API**: Similar to celltypist's interface
- **Automatic Gene Alignment**: Handles gene name matching automatically
- **Pre-trained Models**: Includes full_model trained on complete dataset
- **Rich Visualizations**: Multiple plotting functions for result analysis
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

