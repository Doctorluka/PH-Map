"""
Cell Type Predictor - High-level API for multi-task cell type classification.

This module provides a simplified interface for training and predicting cell types
using the MultiTaskCellTypeClassifier model. It is designed to be similar to 
celltypist's API, with easy-to-use functions for model building and prediction.

Example:
    >>> from phmap.core.predictor import build_model, predict
    >>> 
    >>> # Build a model
    >>> model = build_model(
    ...     adata=reference_data,
    ...     label_columns=['anno_lv1', 'anno_lv2', 'anno_lv3', 'anno_lv4'],
    ...     task_weights=[0.3, 0.8, 1.5, 2.0]
    ... )
    >>> 
    >>> # Save the model
    >>> model.save('models/my_model.pth')
    >>> 
    >>> # Predict on new data (using default model)
    >>> predictions = predict(adata=query_data, return_probabilities=False)
    >>> 
    >>> # Add predictions to AnnData
    >>> query_data.obs = query_data.obs.join(predictions)
"""

import torch
import numpy as np
import pandas as pd
import scanpy as sc
from typing import List, Optional, Dict, Union, Tuple
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import logging

from .classifier import (
    MultiTaskCellTypeClassifier,
    fit_multi_task,
    preprocess_data
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model cache for default model (lazy loading)
_DEFAULT_MODEL_CACHE = None


class PredictionResult:
    """
    Wrapper class for prediction results with probabilities.
    
    This class provides convenient methods to work with prediction results,
    including adding predictions to AnnData objects.
    """
    
    def __init__(self, predictions: pd.DataFrame, probabilities: Dict[str, List[Dict]]):
        """
        Initialize PredictionResult.
        
        Args:
            predictions: DataFrame with predictions and probability values
            probabilities: Dictionary mapping label_column to list of probability dictionaries
        """
        self.predictions = predictions
        self.probabilities = probabilities
    
    def to_adata(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Add predictions to AnnData object's obs.
        
        Args:
            adata: AnnData object to add predictions to
        
        Returns:
            AnnData object with predictions added to obs
        
        Example:
            >>> result = predict(adata=query_data, return_probabilities=True)
            >>> adata = result.to_adata(query_data)
        """
        adata.obs = adata.obs.join(self.predictions)
        return adata


class TrainedModel:
    """
    Wrapper class for a trained MultiTaskCellTypeClassifier model.
    
    This class stores the model along with metadata needed for prediction,
    such as label encoders and gene lists.
    """
    
    def __init__(
        self,
        model: MultiTaskCellTypeClassifier,
        label_columns: List[str],
        label_encoders: Dict[str, LabelEncoder],
        train_genes: List[str],
        var_genes: Optional[List[str]] = None
    ):
        """
        Initialize a TrainedModel wrapper.
        
        Args:
            model: Trained MultiTaskCellTypeClassifier model
            label_columns: List of label column names used during training
            label_encoders: Dictionary mapping label column names to LabelEncoders
            train_genes: List of gene names used during training (in order)
            var_genes: Optional list of variable genes used (if any)
        """
        self.model = model
        self.label_columns = label_columns
        self.label_encoders = label_encoders
        self.train_genes = train_genes
        self.var_genes = var_genes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def save(self, path: str):
        """
        Save the trained model and metadata to a file.
        
        Args:
            path: Path where the model will be saved
        """
        import pickle
        
        # Save the model
        self.model.save(path)
        
        # Save metadata
        metadata_path = path.replace('.pth', '_metadata.pkl')
        metadata = {
            'label_columns': self.label_columns,
            'label_encoders': self.label_encoders,
            'train_genes': self.train_genes,
            'var_genes': self.var_genes
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model and metadata saved to {path} and {metadata_path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        """
        Load a trained model and metadata from files.
        
        Args:
            path: Path to the saved model file
            device: Device to load the model on (default: cuda if available, else cpu)
        
        Returns:
            Loaded TrainedModel instance
        """
        import pickle
        
        # Load the model
        model = MultiTaskCellTypeClassifier.load(path, device=device)
        
        # Load metadata
        metadata_path = path.replace('.pth', '_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        instance = cls(
            model=model,
            label_columns=metadata['label_columns'],
            label_encoders=metadata['label_encoders'],
            train_genes=metadata['train_genes'],
            var_genes=metadata.get('var_genes', None)
        )
        
        logger.info(f"Model and metadata loaded from {path} and {metadata_path}")
        return instance


def build_model(
    adata: sc.AnnData,
    label_columns: List[str],
    task_weights: Optional[List[float]] = None,
    var_genes: Optional[List[str]] = None,
    hidden_sizes: List[int] = [200, 100],
    dropout_rate: float = 0.4,
    batch_size: int = 128,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    device: Optional[torch.device] = None
) -> TrainedModel:
    """
    Build and train a multi-task cell type classifier model.
    
    Args:
        adata: AnnData object containing gene expression data and labels
        label_columns: List of column names in adata.obs containing cell type labels
        task_weights: Optional list of weights for each task (default: [0.3, 0.8, 1.5, 2.0])
        var_genes: Optional list of variable genes to subset the data
        hidden_sizes: List of hidden layer sizes (default: [200, 100])
        dropout_rate: Dropout rate (default: 0.4)
        batch_size: Batch size for training (default: 128)
        num_epochs: Number of training epochs (default: 100)
        learning_rate: Learning rate (default: 0.001)
        early_stopping_patience: Patience for early stopping (default: 10)
        device: Device to train on (default: cuda if available, else cpu)
    
    Returns:
        TrainedModel instance containing the trained model and metadata
    
    Example:
        >>> model = build_model(
        ...     adata=reference_data,
        ...     label_columns=['anno_lv1', 'anno_lv2', 'anno_lv3', 'anno_lv4'],
        ...     task_weights=[0.3, 0.8, 1.5, 2.0]
        ... )
    """
    # Set default task weights if not provided
    if task_weights is None:
        task_weights = [0.3, 0.8, 1.5, 2.0]
    
    # Get number of classes for each task
    num_classes = [
        adata.obs[col].astype('category').cat.categories.size 
        for col in label_columns
    ]
    
    # Get input size
    if var_genes is not None:
        input_size = len(var_genes)
    else:
        input_size = adata.shape[1]
    
    # Create model
    model = MultiTaskCellTypeClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate
    )
    
    # Train model
    fit_multi_task(
        model=model,
        adata=adata,
        label_columns=label_columns,
        num_classes=num_classes,
        task_weights=task_weights,
        var_genes=var_genes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience
    )
    
    # Create label encoders (using training data categories)
    label_encoders = {}
    for label_col in label_columns:
        le = LabelEncoder()
        train_categories = adata.obs[label_col].astype('category').cat.categories
        le.fit(train_categories)
        label_encoders[label_col] = le
    
    # Get gene list used for training
    if var_genes is not None:
        train_genes = var_genes
    else:
        train_genes = list(adata.var_names)
    
    # Create TrainedModel wrapper
    trained_model = TrainedModel(
        model=model,
        label_columns=label_columns,
        label_encoders=label_encoders,
        train_genes=train_genes,
        var_genes=var_genes
    )
    
    logger.info("Model training completed successfully")
    return trained_model


def _prepare_query_data(
    adata: sc.AnnData,
    train_genes: List[str]
) -> np.ndarray:
    """
    Prepare query data for prediction by aligning genes with training data.
    
    Args:
        adata: Query AnnData object
        train_genes: List of gene names used during training (in order)
    
    Returns:
        Binary expression matrix aligned with training genes
    """
    # Get query data expression matrix
    if hasattr(adata.X, 'toarray'):
        query_data = adata.X.toarray()
    else:
        query_data = adata.X
    
    # Create gene name to index mapping for query data
    query_gene_to_idx = {gene: idx for idx, gene in enumerate(adata.var_names)}
    
    # Create aligned expression matrix
    aligned_X = np.zeros((adata.shape[0], len(train_genes)), dtype=np.float32)
    
    for i, gene in enumerate(train_genes):
        if gene in query_gene_to_idx:
            aligned_X[:, i] = query_data[:, query_gene_to_idx[gene]]
    
    # Convert to binary matrix (consistent with training)
    aligned_X_binary = (aligned_X > 0).astype(np.float32)
    
    return aligned_X_binary


def predict(
    model: Optional[Union[TrainedModel, MultiTaskCellTypeClassifier]] = None,
    adata: sc.AnnData = None,
    label_columns: Optional[List[str]] = None,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    train_genes: Optional[List[str]] = None,
    batch_size: int = 128,
    return_probabilities: bool = False,
    device: Optional[torch.device] = None
) -> Union[pd.DataFrame, 'PredictionResult']:
    """
    Predict cell types for query data.
    
    Args:
        model: TrainedModel instance or MultiTaskCellTypeClassifier model.
               If None, automatically loads 'full_model' (default).
        adata: Query AnnData object (required if model is None)
        label_columns: List of label column names (required if model is MultiTaskCellTypeClassifier)
        label_encoders: Dictionary of label encoders (required if model is MultiTaskCellTypeClassifier)
        train_genes: List of training genes (required if model is MultiTaskCellTypeClassifier)
        batch_size: Batch size for prediction (default: 128)
        return_probabilities: Whether to return prediction probabilities (default: False)
        device: Device to use for prediction (default: cuda if available, else cpu)
    
    Returns:
        If return_probabilities=False:
            DataFrame with predictions. Columns are:
            - For each label column: 'predicted_{label_column}'
        
        If return_probabilities=True:
            PredictionResult object containing:
            - predictions: DataFrame with predictions and probability values for predicted labels
              Columns: 'predicted_{label_column}' and 'predicted_{label_column}_prob' (single float value)
            - probabilities: Dictionary mapping label_column to list of probability dictionaries
              Format: {label_column: [dict1, dict2, ...]} where each dict contains all class probabilities
            - to_adata(): Method to add predictions to AnnData object
    
    Example:
        >>> # Using default model (recommended)
        >>> result = predict(adata=query_data, return_probabilities=True)
        >>> 
        >>> # Using custom model
        >>> model = load_pretrained_model('full_model')
        >>> result = predict(model=model, adata=query_data, return_probabilities=True)
        >>> 
        >>> # Without probabilities
        >>> predictions = predict(adata=query_data)
        >>> query_data.obs = query_data.obs.join(predictions)
    """
    global _DEFAULT_MODEL_CACHE
    
    # Handle model parameter: if None, use default model
    if model is None:
        if adata is None:
            raise ValueError("adata is required when model is not provided")
        
        # Lazy load default model
        if _DEFAULT_MODEL_CACHE is None:
            # Import here to avoid circular dependency
            from ..models import load_pretrained_model
            _DEFAULT_MODEL_CACHE = load_pretrained_model('full_model')
            logger.info("Loaded default model 'full_model'")
        model = _DEFAULT_MODEL_CACHE
    
    # Handle different model input types
    if isinstance(model, TrainedModel):
        trained_model = model
        label_columns = trained_model.label_columns
        label_encoders = trained_model.label_encoders
        train_genes = trained_model.train_genes
        pytorch_model = trained_model.model
        device = trained_model.device
    elif isinstance(model, MultiTaskCellTypeClassifier):
        if label_columns is None or label_encoders is None or train_genes is None:
            raise ValueError(
                "If model is MultiTaskCellTypeClassifier, label_columns, "
                "label_encoders, and train_genes must be provided"
            )
        pytorch_model = model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model.to(device)
        pytorch_model.eval()
    else:
        raise TypeError("model must be TrainedModel or MultiTaskCellTypeClassifier")
    
    if adata is None:
        raise ValueError("adata is required")
    
    # Prepare query data
    query_X = _prepare_query_data(adata, train_genes)
    
    # Create DataLoader
    query_dataset = TensorDataset(torch.FloatTensor(query_X))
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
    
    # Perform prediction
    all_predictions = [[] for _ in range(len(label_columns))]
    all_probabilities = [[] for _ in range(len(label_columns))] if return_probabilities else None
    
    pytorch_model.eval()
    with torch.no_grad():
        for batch_X in query_loader:
            batch_X = batch_X[0].to(device)
            outputs = pytorch_model(batch_X)
            
            for i, output in enumerate(outputs):
                # Get predicted classes
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                all_predictions[i].extend(predicted.cpu().numpy())
                
                if return_probabilities:
                    all_probabilities[i].extend(probs.cpu().numpy())
    
    # Convert predictions to label names
    predictions_dict = {}
    all_probabilities_dict = {} if return_probabilities else None
    
    for i, label_col in enumerate(label_columns):
        le = label_encoders[label_col]
        predictions = np.array(all_predictions[i])
        
        # Ensure predictions are within valid range
        valid_indices = (predictions >= 0) & (predictions < len(le.classes_))
        if valid_indices.all():
            predicted_labels = le.inverse_transform(predictions)
        else:
            predicted_labels = np.array(['Unknown'] * len(predictions), dtype=object)
            predicted_labels[valid_indices] = le.inverse_transform(predictions[valid_indices])
            logger.warning(
                f"{label_col}: {(~valid_indices).sum()} invalid prediction indices"
            )
        
        predictions_dict[f'predicted_{label_col}'] = predicted_labels
        
        # Add probabilities if requested
        if return_probabilities:
            probs_array = np.array(all_probabilities[i])
            # Store complete probability dictionaries for all classes
            prob_dicts = []
            # Store single probability values for predicted labels (for DataFrame)
            predicted_probs = []
            
            for j, prob_vec in enumerate(probs_array):
                # Create complete probability dictionary for all classes
                prob_dict = {
                    le.classes_[k]: float(prob_vec[k]) 
                    for k in range(len(le.classes_))
                }
                prob_dicts.append(prob_dict)
                
                # Extract probability value for the predicted label only
                predicted_idx = predictions[j]
                if predicted_idx >= 0 and predicted_idx < len(le.classes_):
                    predicted_probs.append(float(prob_vec[predicted_idx]))
                else:
                    predicted_probs.append(0.0)
            
            # Store single probability value in DataFrame
            predictions_dict[f'predicted_{label_col}_prob'] = predicted_probs
            # Store complete probability dictionaries separately
            all_probabilities_dict[label_col] = prob_dicts
    
    # Create DataFrame
    result_df = pd.DataFrame(predictions_dict, index=adata.obs.index)
    
    logger.info(f"Prediction completed for {len(result_df)} cells")
    
    # Return format depends on whether probabilities are requested
    if return_probabilities:
        return PredictionResult(
            predictions=result_df,
            probabilities=all_probabilities_dict
        )
    else:
        return result_df


def evaluate(
    predictions: pd.DataFrame,
    true_labels: pd.DataFrame,
    label_columns: List[str]
) -> Dict[str, Dict]:
    """
    Evaluate prediction results against true labels.
    
    Args:
        predictions: DataFrame with predicted labels (from predict function)
        true_labels: DataFrame with true labels (should have same index as predictions)
        label_columns: List of label column names to evaluate
    
    Returns:
        Dictionary containing evaluation metrics for each task:
        - 'accuracy': Overall accuracy
        - 'classification_report': Detailed classification report (as dict)
        - 'confusion_matrix': Confusion matrix (as numpy array)
    
    Example:
        >>> true_labels = query_data.obs[['anno_lv1', 'anno_lv2', 'anno_lv4']]
        >>> results = evaluate(predictions, true_labels, ['anno_lv1', 'anno_lv2', 'anno_lv4'])
        >>> print(results['anno_lv4']['accuracy'])
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix
    )
    
    results = {}
    
    for label_col in label_columns:
        pred_col = f'predicted_{label_col}'
        
        if pred_col not in predictions.columns:
            logger.warning(f"Prediction column {pred_col} not found, skipping {label_col}")
            continue
        
        if label_col not in true_labels.columns:
            logger.warning(f"True label column {label_col} not found, skipping")
            continue
        
        # Align indices
        common_idx = predictions.index.intersection(true_labels.index)
        if len(common_idx) == 0:
            logger.warning(f"No common indices found for {label_col}, skipping")
            continue
        
        pred = predictions.loc[common_idx, pred_col]
        true = true_labels.loc[common_idx, label_col]
        
        # Calculate accuracy
        accuracy = accuracy_score(true, pred)
        
        # Generate classification report
        report = classification_report(
            true, pred, 
            output_dict=True,
            zero_division=0
        )
        
        # Generate confusion matrix
        # Get all unique labels
        all_labels = sorted(set(true.unique()) | set(pred.unique()))
        cm = confusion_matrix(true, pred, labels=all_labels)
        
        results[label_col] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'confusion_matrix_labels': all_labels
        }
        
        logger.info(f"{label_col} accuracy: {accuracy:.4f}")
    
    return results

