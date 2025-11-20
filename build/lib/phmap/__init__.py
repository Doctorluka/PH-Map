"""
PH-Map: Multi-task cell type classification package for single-cell RNA sequencing data.

This package provides a simplified API for multi-task cell type classification,
similar to celltypist's usage pattern.

Example:
    >>> import phmap
    >>> import scanpy as sc
    >>> 
    >>> # Load query data
    >>> adata = sc.read_h5ad('query_data.h5ad')
    >>> 
    >>> # Predict using default model (recommended)
    >>> result = phmap.predict(adata, return_probabilities=True)
    >>> 
    >>> # Visualize
    >>> phmap.pl.plot_probability_bar(result, label_columns=['anno_lv4'])
    >>> 
    >>> # Add to AnnData
    >>> adata = result.to_adata(adata)
"""

# Print startup information (logo, version, GPU info)
from ._startup import _print_startup_info, _PRINT_STARTUP
if _PRINT_STARTUP:
    _print_startup_info()

# 核心功能
from .core.predictor import (
    build_model,
    predict,
    evaluate,
    TrainedModel,
    PredictionResult
)

# 预训练模型加载
from .models import load_pretrained_model, list_available_models

# 版本信息
from .version import __version__

# Plotting模块（作为子模块）
from . import pl

__all__ = [
    'build_model',
    'predict',
    'evaluate',
    'TrainedModel',
    'PredictionResult',
    'load_pretrained_model',
    'list_available_models',
    '__version__',
    'pl',  # plotting模块
]

# Export startup control functions (optional, for advanced users)
from ._startup import _suppress_startup, _enable_startup
__all__.extend(['_suppress_startup', '_enable_startup'])

