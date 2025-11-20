import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import logging
import scanpy as sc
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneExpressionDataset(Dataset):
    """Dataset class for gene expression data (with multiple tasks)."""
    def __init__(self, X: np.ndarray, y: List[np.ndarray]):
        self.X = torch.FloatTensor(X)
        self.y = [torch.LongTensor(label) for label in y]  # List of task labels
    
    def __len__(self):
        return len(self.y[0])  # All tasks should have the same number of samples
    
    def __getitem__(self, idx):
        return self.X[idx], [y[idx] for y in self.y]


# 修改后的 MultiTaskCellTypeClassifier 类，支持多任务学习
class MultiTaskCellTypeClassifier(torch.nn.Module):
    """Multi-task neural network model for cell type classification."""
    def __init__(self, input_size: int, num_classes: List[int], 
                 hidden_sizes: List[int] = [200, 100], 
                 dropout_rate: float = 0.4):
        super().__init__()
        
        # Save hyperparameters for model saving/loading
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        layers = []
        prev_size = input_size
        
        # Build shared hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate)
            ])
            prev_size = hidden_size
        
        # Shared layers for all tasks
        self.shared_fc = torch.nn.Sequential(*layers)

        # Output layers for each task (LV1, LV2, LV3, LV4)
        self.output_layers = torch.nn.ModuleList([
            torch.nn.Linear(prev_size, num_class) for num_class in num_classes
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        shared_features = self.shared_fc(x)
        return [output_layer(shared_features) for output_layer in self.output_layers]

    def compute_loss(self, outputs: List[torch.Tensor], labels: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        losses = [torch.nn.CrossEntropyLoss()(output, label) for output, label in zip(outputs, labels)]
        weighted_loss = sum([w * loss for w, loss in zip(weights, losses)])
        return weighted_loss  # Sum of weighted losses from all tasks

    def print_training_parameters(self):
        """Print training parameters and model configurations."""
        print("=" * 60)
        print("MultiTaskCellTypeClassifier Training Parameters")
        print("=" * 60)
        # You can extend this method to print hyperparameters like learning rate, dropout, etc.
        print("=" * 60)
    
    def save(self, path: str):
        """Save the model to a file.
        
        Args:
            path: Path where the model will be saved
        
        Example:
            >>> model = MultiTaskCellTypeClassifier(
            ...     input_size=2000,
            ...     num_classes=[10, 20, 30, 50],
            ...     hidden_sizes=[200, 100],
            ...     dropout_rate=0.4
            ... )
            >>> # Train the model...
            >>> model.save("models/cell_type_classifier.pth")
        """
        # Save both model state_dict and hyperparameters for easy loading
        save_dict = {
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        """Load a model from a file.
        
        Args:
            path: Path to the saved model file
            device: Device to load the model on (default: cuda if available, else cpu)
        
        Returns:
            Loaded MultiTaskCellTypeClassifier model
        
        Example:
            >>> # Load model (automatically uses CUDA if available)
            >>> model = MultiTaskCellTypeClassifier.load("models/cell_type_classifier.pth")
            >>> 
            >>> # Load model to specific device
            >>> device = torch.device("cpu")
            >>> model = MultiTaskCellTypeClassifier.load("models/cell_type_classifier.pth", device=device)
            >>> 
            >>> # Use the loaded model for inference
            >>> model.eval()
            >>> with torch.no_grad():
            ...     outputs = model(input_tensor)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        
        # Reconstruct model from saved hyperparameters
        model = cls(
            input_size=checkpoint['input_size'],
            num_classes=checkpoint['num_classes'],
            hidden_sizes=checkpoint['hidden_sizes'],
            dropout_rate=checkpoint['dropout_rate']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model

# 在 fit 函数中使用多任务学习
def fit_multi_task(
    model,
    adata: sc.AnnData,
    label_columns: List[str],
    num_classes: List[int],
    task_weights: List[float] = [0.3, 0.8, 1.5, 2.0],
    var_genes: Optional[List[str]] = None,
    batch_size: int = 128,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
):
    """
    Train the model with multi-task learning for multiple label columns (e.g., LV1, LV2, LV3, final_celltype).

    Args:
        model: The model to be trained (MultiTaskCellTypeClassifier)
        adata: AnnData object containing gene expression data
        label_columns: List of column names in adata.obs containing cell type labels
        num_classes: List containing the number of classes for each task
        var_genes: Optional list of variable genes to subset the data
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        early_stopping_patience: Patience for early stopping
    """
    # Ensure task_weights matches the number of classes
    if len(task_weights) != len(num_classes):
        raise ValueError("Length of task_weights must match the number of tasks (num_classes)")

    # Preprocess the data (convert to binary matrix and encode labels)
    X, y = preprocess_data(adata, label_columns, var_genes)

    # Ensure X is a dense numpy array (convert from possible scipy sparse matrix)
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # Prepare DataLoader
    train_loader, val_loader = prepare_data_loaders(X, y, batch_size)

    # Select device (cuda if available, otherwise cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0
    patience_counter = 0
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Initialize train_correct and train_total for each task at the start of the epoch
        train_correct = [0] * len(num_classes)
        train_total = [0] * len(num_classes)

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = [y.to(device) for y in batch_y]

            # Get model outputs for each task
            outputs = model(batch_X)

            # Compute loss for each task
            loss = model.compute_loss(outputs, batch_y, task_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy for each task
            for i in range(len(outputs)):
                _, predicted = torch.max(outputs[i], 1)
                train_correct[i] += (predicted == batch_y[i]).sum().item()
                train_total[i] += batch_y[i].size(0)

        # Calculate task-wise accuracy for this epoch
        task_accuracies = [
            100.0 * train_correct[i] / train_total[i] if train_total[i] > 0 else 0.0
            for i in range(len(outputs))
        ]
        train_accuracies.append(np.mean(task_accuracies))

        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_loss:.4f}")
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Accuracy: {np.mean(task_accuracies):.2f}%")

        # Validation phase
        model.eval()

        val_correct = [0] * len(num_classes)
        val_total = [0] * len(num_classes)

        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = [y.to(device) for y in batch_y]
            outputs = model(batch_X)

            for i in range(len(outputs)):
                _, predicted = torch.max(outputs[i], 1)
                val_correct[i] += (predicted == batch_y[i]).sum().item()
                val_total[i] += batch_y[i].size(0)

        task_accuracies_val = [
            100.0 * val_correct[i] / val_total[i] if val_total[i] > 0 else 0.0
            for i in range(len(outputs))
        ]
        val_accuracies.append(task_accuracies_val)

        # Calculate average validation accuracy across tasks
        val_avg_acc = np.mean(task_accuracies_val)

        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {val_avg_acc:.2f}%")

        # Early stopping
        if val_avg_acc > best_val_accuracy:
            best_val_accuracy = val_avg_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")


# 数据预处理函数（将数据转为二值化矩阵并进行标签编码）
def preprocess_data(adata: sc.AnnData, label_columns: List[str], var_genes: Optional[List[str]] = None):
    """Preprocess the AnnData object for multi-task classification."""
    if var_genes is not None:
        adata = adata[:, list(var_genes)].copy()
    else:
        var_genes = list(adata.var_names)

    # Convert counts to binary matrix (1 if gene is expressed, else 0)
    X = (adata.X > 0).astype(np.float32)
    
    # Prepare the labels for all tasks
    y = []
    for label_column in label_columns:
        y_label = adata.obs[label_column].values
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_label)
        y.append(y_encoded)
    
    # Return X (features) and y (list of encoded labels for each task)
    return X, y


# 数据加载器准备函数
def prepare_data_loaders(X: np.ndarray, y: list, batch_size: int):
    """Prepare DataLoader for training and validation for multi-task labels.

    Args:
        X: Feature matrix (n_samples, n_genes)
        y: List of label arrays, one per task. Each shape (n_samples,)
        batch_size: Mini-batch size

    Returns:
        train_loader, val_loader: DataLoader for training and validation
    """
    # Convert y (list of np arrays, shape (n_samples,)) to shape (n_samples, n_tasks)
    y_stacked = np.stack(y, axis=1)  # shape: (n_samples, n_tasks)
    # Use the first task for stratification, or use no stratification if you want to avoid errors
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_stacked, test_size=0.2, random_state=42, stratify=y_stacked[:, 0]
    )
    # For multi-task, convert back into list-of-arrays format
    y_train_list = [y_train[:, i] for i in range(y_train.shape[1])]
    y_val_list = [y_val[:, i] for i in range(y_val.shape[1])]

    train_dataset = GeneExpressionDataset(X_train, y_train_list)
    val_dataset = GeneExpressionDataset(X_val, y_val_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

