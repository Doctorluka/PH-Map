"""
Plotting utilities for cell type prediction results.

This module provides functions to visualize prediction results, including
probability heatmaps, bar charts, and cell type counts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import List, Optional, Dict, Union
import scanpy as sc


def plot_probability_heatmap(
    result: 'PredictionResult',
    label_columns: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
    cmap: str = 'viridis',
    show_cbar: bool = True,
    show: bool = True,
    ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None
):
    """
    Plot probability heatmap for each level.

    Shows the average prediction probability for each predicted cell type.
    For each cell type, calculates the mean probability of cells that were predicted as that type.
    This helps assess prediction confidence: higher values indicate more confident predictions.

    Args:
        result: PredictionResult object from predict() function
        label_columns: List of label columns to plot (default: all)
        figsize: Figure size (width, height) - only used if ax is None
        cmap: Colormap for heatmap
        show_cbar: Whether to show colorbar
        show: Whether to call plt.show()
        ax: Matplotlib axes object(s) to plot on. If None, creates new figure.
            Can be a single Axes or a list of Axes (one per label_column).
            If provided, figsize is ignored.

    Returns:
        Matplotlib axes object(s). Single Axes if one label_column, list of Axes if multiple.

    Example:
        >>> result = predict(model=trained_model, adata=query_data, return_probabilities=True)
        >>> # Create new figure
        >>> ax = plot_probability_heatmap(result, label_columns=['final_celltype'])
        >>> 
        >>> # Use existing axes
        >>> fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        >>> plot_probability_heatmap(result, label_columns=['final_celltype'], ax=ax)
    """
    if label_columns is None:
        # Extract label columns from predictions DataFrame
        pred_cols = [col for col in result.predictions.columns if col.startswith('predicted_')]
        label_columns = [col.replace('predicted_', '') for col in pred_cols if '_prob' not in col]

    n_levels = len(label_columns)
    
    # Handle ax parameter
    if ax is None:
        fig, axes = plt.subplots(1, n_levels, figsize=(figsize[0] * n_levels, figsize[1]))
        if n_levels == 1:
            axes = [axes]
        return_fig = True
    else:
        # Use provided axes
        if n_levels == 1:
            axes = [ax] if not isinstance(ax, list) else ax
        else:
            axes = ax if isinstance(ax, list) else [ax]
        if len(axes) != n_levels:
            raise ValueError(f"Number of axes ({len(axes)}) must match number of label_columns ({n_levels})")
        return_fig = False

    for idx, label_col in enumerate(label_columns):
        pred_col = f'predicted_{label_col}'
        prob_col = f'predicted_{label_col}_prob'

        if pred_col not in result.predictions.columns or prob_col not in result.predictions.columns:
            print(f"Warning: {pred_col} or {prob_col} not found in predictions, skipping...")
            continue

        # Group by predicted label and calculate mean probability for each predicted cell type
        mean_probs = result.predictions.groupby(pred_col)[prob_col].mean().sort_values(ascending=False)

        # Create heatmap data (single row for mean probabilities)
        heatmap_data = mean_probs.values.reshape(1, -1)

        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            yticklabels=['Mean Prediction Probability'],
            xticklabels=mean_probs.index,
            cmap=cmap,
            cbar=show_cbar,
            ax=axes[idx],
            annot=True,
            fmt='.3f'
        )
        axes[idx].set_title(f'Prediction Probability Heatmap - {label_col}', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Predicted Cell Type', fontsize=10)
        plt.setp(axes[idx].get_xticklabels(), rotation=90, ha='right')

    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
    
    # Return axes (single or list)
    return axes[0] if n_levels == 1 else axes


def optim_palette(
    adata: sc.AnnData,
    obs_columns: List[str],
    color_scheme: str = "Paired",
) -> Union[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Generate optimized categorical color palettes based on obs columns.

    This function is inspired by the R pattern:
        library(RColorBrewer)
        getPalette = colorRampPalette(brewer.pal(12, "Paired"))
        mycolors <- getPalette(n)

    It collects all categories across the specified obs columns, assigns
    a unique color to each category using a matplotlib qualitative colormap.

    The same category label appearing in multiple obs columns will receive
    the same color.

    Args:
        adata: AnnData object with categorical columns in .obs.
        obs_columns: List of obs column names (string or categorical type).
        color_scheme: Name of qualitative color scheme.
            Default: "Paired".
            Supported (built-in): "Paired", "Set3", "Dark2", "tab20", "Accent".
            Any valid matplotlib colormap name is also accepted.

    Returns:
        If a single obs_column is provided:
            Dict[str, str]: {category: "#RRGGBB", ...}
        
        If multiple obs_columns are provided:
            Dict[str, Dict[str, str]]: {obs_col: {category: "#RRGGBB", ...}, ...}
    """
    if isinstance(obs_columns, str):
        obs_columns = [obs_columns]

    # Supported qualitative schemes (matplotlib names)
    qualitative_schemes = {
        "Paired": "Paired",   # default, similar to RColorBrewer::Paired
        "Set3": "Set3",
        "Dark2": "Dark2",
        "tab20": "tab20",
        "Accent": "Accent",
    }

    cmap_name = qualitative_schemes.get(color_scheme, color_scheme)
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError as e:
        raise ValueError(
            f"Unknown color_scheme '{color_scheme}'. "
            f"Supported schemes: {list(qualitative_schemes.keys())} "
            f"or any valid matplotlib colormap name."
        ) from e

    # Collect categories for each column and global unique category list
    col_to_categories: Dict[str, List[str]] = {}
    all_categories: List[str] = []

    for col in obs_columns:
        if col not in adata.obs.columns:
            raise KeyError(f"Column '{col}' not found in adata.obs")

        series = adata.obs[col]
        if not pd.api.types.is_categorical_dtype(series):
            series = series.astype("category")

        cats = list(series.cat.categories)
        col_to_categories[col] = cats
        all_categories.extend(cats)

    # Deduplicate while preserving order
    seen = set()
    unique_categories: List[str] = []
    for cat in all_categories:
        if cat not in seen:
            seen.add(cat)
            unique_categories.append(cat)

    n = len(unique_categories)
    if n == 0:
        if len(obs_columns) == 1:
            return {}
        return {col: {} for col in obs_columns}

    if n == 1:
        positions = [0.5]
    else:
        positions = np.linspace(0.0, 1.0, n)

    colors = [mcolors.to_hex(cmap(pos)) for pos in positions]
    category_to_color = {
        cat: color for cat, color in zip(unique_categories, colors)
    }

    # Build nested mapping per obs column
    result: Dict[str, Dict[str, str]] = {}
    for col, cats in col_to_categories.items():
        result[col] = {cat: category_to_color[cat] for cat in cats}

    # If only one obs_column, return the inner dict directly for convenience
    if len(obs_columns) == 1:
        return result[obs_columns[0]]
    
    return result

def plot_probability_bar(
    result: 'PredictionResult',
    label_columns: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    figsize: tuple = (10, 6),
    orientation: str = 'vertical',
    show: bool = True,
    ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None
):
    """
    Plot probability bar chart for each level.

    Shows the average prediction probability for each predicted cell type (or top N types).
    For each cell type, calculates the mean probability of cells that were predicted as that type.
    This helps assess prediction confidence: higher values indicate more confident predictions.

    Args:
        result: PredictionResult object from predict() function
        label_columns: List of label columns to plot (default: all)
        top_n: Number of top cell types to show (default: all)
        figsize: Figure size (width, height) - only used if ax is None
        orientation: 'vertical' or 'horizontal'
        show: Whether to call plt.show()
        ax: Matplotlib axes object(s) to plot on. If None, creates new figure.
            Can be a single Axes or a list of Axes (one per label_column).
            If provided, figsize is ignored.

    Returns:
        Matplotlib axes object(s). Single Axes if one label_column, list of Axes if multiple.

    Example:
        >>> result = predict(model=trained_model, adata=query_data, return_probabilities=True)
        >>> # Create new figure
        >>> ax = plot_probability_bar(result, label_columns=['final_celltype'], top_n=20)
        >>> 
        >>> # Use existing axes
        >>> fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        >>> plot_probability_bar(result, label_columns=['final_celltype'], ax=ax)
    """
    if label_columns is None:
        # Extract label columns from predictions DataFrame
        pred_cols = [col for col in result.predictions.columns if col.startswith('predicted_')]
        label_columns = [col.replace('predicted_', '') for col in pred_cols if '_prob' not in col]

    n_levels = len(label_columns)
    
    # Handle ax parameter
    if ax is None:
        fig, axes = plt.subplots(1, n_levels, figsize=(figsize[0] * n_levels, figsize[1]))
        if n_levels == 1:
            axes = [axes]
        return_fig = True
    else:
        # Use provided axes
        if n_levels == 1:
            axes = [ax] if not isinstance(ax, list) else ax
        else:
            axes = ax if isinstance(ax, list) else [ax]
        if len(axes) != n_levels:
            raise ValueError(f"Number of axes ({len(axes)}) must match number of label_columns ({n_levels})")
        return_fig = False

    for idx, label_col in enumerate(label_columns):
        pred_col = f'predicted_{label_col}'
        prob_col = f'predicted_{label_col}_prob'

        if pred_col not in result.predictions.columns or prob_col not in result.predictions.columns:
            print(f"Warning: {pred_col} or {prob_col} not found in predictions, skipping...")
            continue

        # Group by predicted label and calculate mean probability for each predicted cell type
        mean_probs = result.predictions.groupby(pred_col)[prob_col].mean().sort_values(ascending=False)

        # Select top N if specified
        if top_n is not None:
            mean_probs = mean_probs.head(top_n)

        # Plot bar chart
        if orientation == 'vertical':
            axes[idx].bar(range(len(mean_probs)), mean_probs.values)
            axes[idx].set_xticks(range(len(mean_probs)))
            axes[idx].set_xticklabels(mean_probs.index, rotation=90, ha='right')
            axes[idx].set_ylabel('Mean Prediction Probability', fontsize=10)
            axes[idx].set_xlabel('Predicted Cell Type', fontsize=10)
        else:  # horizontal
            axes[idx].barh(range(len(mean_probs)), mean_probs.values)
            axes[idx].set_yticks(range(len(mean_probs)))
            axes[idx].set_yticklabels(mean_probs.index)
            axes[idx].set_xlabel('Mean Prediction Probability', fontsize=10)
            axes[idx].set_ylabel('Predicted Cell Type', fontsize=10)

        axes[idx].set_title(f'Prediction Probability Bar Chart - {label_col}', fontsize=10, fontweight='bold')
        axes[idx].grid(axis='y' if orientation == 'vertical' else 'x', alpha=0.3)

    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
    
    # Return axes (single or list)
    return axes[0] if n_levels == 1 else axes


def plot_cell_type_counts(
    result: 'PredictionResult',
    label_columns: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    figsize: tuple = (10, 6),
    orientation: str = 'vertical',
    show: bool = True,
    ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None
):
    """
    Plot cell type count bar chart for each level.

    Shows the number of cells for each predicted cell type.

    Args:
        result: PredictionResult object from predict() function
        label_columns: List of label columns to plot (default: all)
        top_n: Number of top cell types to show (default: all)
        figsize: Figure size (width, height) - only used if ax is None
        orientation: 'vertical' or 'horizontal'
        show: Whether to call plt.show()
        ax: Matplotlib axes object(s) to plot on. If None, creates new figure.
            Can be a single Axes or a list of Axes (one per label_column).
            If provided, figsize is ignored.

    Returns:
        Matplotlib axes object(s). Single Axes if one label_column, list of Axes if multiple.

    Example:
        >>> result = predict(model=trained_model, adata=query_data, return_probabilities=True)
        >>> # Create new figure
        >>> ax = plot_cell_type_counts(result, label_columns=['final_celltype'], top_n=20)
        >>> 
        >>> # Use existing axes
        >>> fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        >>> plot_cell_type_counts(result, label_columns=['final_celltype'], ax=ax)
    """
    if label_columns is None:
        # Extract label columns from predictions DataFrame
        pred_cols = [col for col in result.predictions.columns if col.startswith('predicted_')]
        label_columns = [col.replace('predicted_', '') for col in pred_cols if '_prob' not in col]

    n_levels = len(label_columns)
    
    # Handle ax parameter
    if ax is None:
        fig, axes = plt.subplots(1, n_levels, figsize=(figsize[0] * n_levels, figsize[1]))
        if n_levels == 1:
            axes = [axes]
        return_fig = True
    else:
        # Use provided axes
        if n_levels == 1:
            axes = [ax] if not isinstance(ax, list) else ax
        else:
            axes = ax if isinstance(ax, list) else [ax]
        if len(axes) != n_levels:
            raise ValueError(f"Number of axes ({len(axes)}) must match number of label_columns ({n_levels})")
        return_fig = False

    for idx, label_col in enumerate(label_columns):
        pred_col = f'predicted_{label_col}'

        if pred_col not in result.predictions.columns:
            print(f"Warning: {pred_col} not found in predictions, skipping...")
            continue

        # Count cells for each cell type
        counts = result.predictions[pred_col].value_counts().sort_values(ascending=False)

        # Select top N if specified
        if top_n is not None:
            counts = counts.head(top_n)

        # Plot bar chart
        if orientation == 'vertical':
            axes[idx].bar(range(len(counts)), counts.values)
            axes[idx].set_xticks(range(len(counts)))
            axes[idx].set_xticklabels(counts.index, rotation=90, ha='right')
            axes[idx].set_ylabel('Number of Cells', fontsize=10)
            axes[idx].set_xlabel('Cell Type', fontsize=10)
        else:  # horizontal
            axes[idx].barh(range(len(counts)), counts.values)
            axes[idx].set_yticks(range(len(counts)))
            axes[idx].set_yticklabels(counts.index)
            axes[idx].set_xlabel('Number of Cells', fontsize=10)
            axes[idx].set_ylabel('Cell Type', fontsize=10)

        axes[idx].set_title(f'Cell Type Counts - {label_col}', fontsize=10, fontweight='bold')
        axes[idx].grid(axis='y' if orientation == 'vertical' else 'x', alpha=0.3)

    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
    
    # Return axes (single or list)
    return axes[0] if n_levels == 1 else axes


def plot_probability_distribution(
    result: 'PredictionResult',
    label_columns: Optional[List[str]] = None,
    figsize: tuple = (10, 6),
    show: bool = True,
    ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None
):
    """
    Plot probability distribution for each level.

    Shows the distribution of prediction probabilities across all cells.

    Args:
        result: PredictionResult object from predict() function
        label_columns: List of label columns to plot (default: all)
        figsize: Figure size (width, height) - only used if ax is None
        show: Whether to call plt.show()
        ax: Matplotlib axes object(s) to plot on. If None, creates new figure.
            Can be a single Axes or a list of Axes (one per label_column).
            If provided, figsize is ignored.

    Returns:
        Matplotlib axes object(s). Single Axes if one label_column, list of Axes if multiple.

    Example:
        >>> result = predict(model=trained_model, adata=query_data, return_probabilities=True)
        >>> # Create new figure
        >>> ax = plot_probability_distribution(result, label_columns=['final_celltype'])
        >>> 
        >>> # Use existing axes
        >>> fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        >>> plot_probability_distribution(result, label_columns=['final_celltype'], ax=ax)
    """
    if label_columns is None:
        label_columns = list(result.probabilities.keys())

    n_levels = len(label_columns)
    
    # Handle ax parameter
    if ax is None:
        fig, axes = plt.subplots(1, n_levels, figsize=(figsize[0] * n_levels, figsize[1]))
        if n_levels == 1:
            axes = [axes]
        return_fig = True
    else:
        # Use provided axes
        if n_levels == 1:
            axes = [ax] if not isinstance(ax, list) else ax
        else:
            axes = ax if isinstance(ax, list) else [ax]
        if len(axes) != n_levels:
            raise ValueError(f"Number of axes ({len(axes)}) must match number of label_columns ({n_levels})")
        return_fig = False

    for idx, label_col in enumerate(label_columns):
        prob_col = f'predicted_{label_col}_prob'

        if prob_col not in result.predictions.columns:
            print(f"Warning: {prob_col} not found in predictions, skipping...")
            continue

        # Plot histogram
        axes[idx].hist(result.predictions[prob_col].values, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel('Prediction Probability', fontsize=10)
        axes[idx].set_ylabel('Number of Cells', fontsize=10)
        axes[idx].set_title(f'Probability Distribution - {label_col}', fontsize=10, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

        # Add statistics
        mean_prob = result.predictions[prob_col].mean()
        median_prob = result.predictions[prob_col].median()
        axes[idx].axvline(mean_prob, color='red', linestyle='--', label=f'Mean: {mean_prob:.3f}')
        axes[idx].axvline(median_prob, color='green', linestyle='--', label=f'Median: {median_prob:.3f}')
        axes[idx].legend()

    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
    
    # Return axes (single or list)
    return axes[0] if n_levels == 1 else axes

