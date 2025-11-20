"""
模型注册表，管理可用的预训练模型
"""
import os
from pathlib import Path
from typing import Dict, Optional

# 获取包内模型目录
try:
    from importlib import resources
    _USE_RESOURCES = True
except ImportError:
    try:
        import pkg_resources
        _USE_RESOURCES = False
    except ImportError:
        _USE_RESOURCES = None

def _get_model_path(model_name: str) -> Optional[Path]:
    """获取模型文件路径"""
    model_file = MODEL_REGISTRY[model_name]['model_file']
    
    # Always use __file__ approach for subdirectory files
    # importlib.resources.path() doesn't support subdirectories
    package_dir = Path(__file__).parent
    model_path = package_dir / model_file
    
    if model_path.exists():
        return model_path
    
    # Try alternative approaches if file not found
    if _USE_RESOURCES:
        # Python 3.9+ approach with files() API
        try:
            from importlib.resources import files as resource_files
            model_path = resource_files('phmap.models') / model_file
            if model_path.exists():
                return Path(str(model_path))
        except (ModuleNotFoundError, TypeError, AttributeError):
            pass
    
    if not _USE_RESOURCES:
        # pkg_resources approach (older Python versions)
        try:
            resource_path = pkg_resources.resource_filename('phmap.models', model_file)
            if Path(resource_path).exists():
                return Path(resource_path)
        except Exception:
            pass
    
    # Final fallback
    return model_path

# 模型注册表
MODEL_REGISTRY = {
    'full_model': {
        'name': 'full_model',
        'description': 'Full model trained on complete dataset',
        'model_file': 'full_model/cell_type_classifier_full.pth',
        'metadata_file': 'full_model/cell_type_classifier_full_metadata.pkl',
        'label_columns': ['anno_lv1', 'anno_lv2', 'anno_lv3', 'anno_lv4'],
    },
}

def list_available_models() -> Dict:
    """列出所有可用的预训练模型"""
    return MODEL_REGISTRY.copy()

def get_model_path(model_name: str) -> Optional[Path]:
    """获取模型文件路径"""
    if model_name not in MODEL_REGISTRY:
        return None
    return _get_model_path(model_name)

def get_metadata_path(model_name: str) -> Optional[Path]:
    """获取元数据文件路径"""
    if model_name not in MODEL_REGISTRY:
        return None
    
    metadata_file = MODEL_REGISTRY[model_name]['metadata_file']
    
    # Always use __file__ approach for subdirectory files
    package_dir = Path(__file__).parent
    metadata_path = package_dir / metadata_file
    
    if metadata_path.exists():
        return metadata_path
    
    # Try alternative approaches if file not found
    if _USE_RESOURCES:
        try:
            from importlib.resources import files as resource_files
            metadata_path = resource_files('phmap.models') / metadata_file
            if metadata_path.exists():
                return Path(str(metadata_path))
        except (ModuleNotFoundError, TypeError, AttributeError):
            pass
    
    if not _USE_RESOURCES:
        try:
            resource_path = pkg_resources.resource_filename('phmap.models', metadata_file)
            if Path(resource_path).exists():
                return Path(resource_path)
        except Exception:
            pass
    
    # Final fallback
    return metadata_path

