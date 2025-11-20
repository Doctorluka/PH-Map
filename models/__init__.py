"""
模型管理模块 - 提供预训练模型的加载功能
"""
from ..core.predictor import TrainedModel
from .__model_registry__ import (
    MODEL_REGISTRY,
    list_available_models,
    get_model_path,
    get_metadata_path
)

def load_pretrained_model(model_name: str = 'full_model', **kwargs):
    """
    加载预训练模型
    
    Args:
        model_name: 模型名称（如'full_model'）或模型文件路径（.pth文件）
                   - 如果是注册表中的名称（如'full_model'），从包内资源加载
                   - 如果是文件路径，从指定路径加载
        **kwargs: 传递给TrainedModel.load的其他参数（如device）
    
    Returns:
        TrainedModel实例
    
    Example:
        >>> import phmap
        >>> # 加载内置模型
        >>> model = phmap.load_pretrained_model('full_model')
        >>> 
        >>> # 从文件路径加载
        >>> model = phmap.load_pretrained_model('path/to/my_model.pth')
        >>> 
        >>> # 指定设备
        >>> import torch
        >>> model = phmap.load_pretrained_model('full_model', device=torch.device('cpu'))
    """
    from pathlib import Path
    
    # Check if model_name is a file path (ends with .pth)
    model_path_obj = Path(model_name)
    if model_name.endswith('.pth') and model_path_obj.exists():
        # It's a file path, load directly
        return TrainedModel.load(str(model_path_obj), **kwargs)
    
    # Otherwise, treat it as a registered model name
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available built-in models: {available}. "
            f"If you want to load from a file, provide a path to a .pth file."
        )
    
    # Get path for built-in model
    model_path = get_model_path(model_name)
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            f"Built-in model file not found for '{model_name}'. "
            f"Expected path: {model_path}. "
            f"Please ensure the package is properly installed."
        )
    
    return TrainedModel.load(str(model_path), **kwargs)

__all__ = ['load_pretrained_model', 'list_available_models']

