"""
PH-Map startup information and GPU detection.
"""
import sys
from typing import Optional, Tuple

def _get_gpu_info() -> Tuple[bool, Optional[dict]]:
    """
    Detect GPU and CUDA information.
    
    Returns:
        Tuple of (cuda_available, gpu_info_dict)
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if not cuda_available:
            return False, None
        
        gpu_info = {
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
            'compute_capability': f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
        }
        
        return True, gpu_info
    except ImportError:
        return False, None
    except Exception:
        return False, None


def _print_startup_info():
    """Print startup information including logo, version, and GPU info."""
    # ASCII Art Logo for PHMAP (complete)
    logo = """
   ____  _   __  __    _    ____  
  |  _ \| | |  \/  |  / \  |  _ \ 
  | |_) | |_| |\/| | / _ \ | |_) |
  |  __/|  _  |  | |/ ___ \|  __/ 
  |_|   |_| |_|  |_/_/   \_\_|    
                                  
  🔬 Multi-task Learning Framework for Cell Type Classifier
  🫁 pretrained model on pulmonary hypertension scRNA-seq data
    """
    
    # Get version (use relative import to avoid circular dependency)
    try:
        from .version import __version__
    except ImportError:
        # Fallback for direct execution
        from pathlib import Path
        version_file = Path(__file__).parent / 'version.py'
        version_dict = {}
        exec(open(version_file).read(), version_dict)
        __version__ = version_dict['__version__']
    
    # Get GPU info
    cuda_available, gpu_info = _get_gpu_info()
    
    # Print logo and version
    print(logo)
    print(f"🔖 Version: {__version__}")
    print()
    
    # Print GPU information
    print("🧬 Detecting CUDA devices…")
    if cuda_available and gpu_info:
        print(f"✅ [GPU {gpu_info['current_device']}] {gpu_info['device_name']}")
        print(f"   • Total memory: {gpu_info['memory_total']:.1f} GB")
        print(f"   • Compute capability: {gpu_info['compute_capability']}")
        if gpu_info.get('cuda_version'):
            print(f"   • CUDA version: {gpu_info['cuda_version']}")
        if gpu_info['device_count'] > 1:
            print(f"   • Total GPUs: {gpu_info['device_count']}")
    else:
        print("⚠️  No GPU detected, using CPU")
        try:
            import torch
            if not torch.cuda.is_available():
                print("   • PyTorch CUDA support: Not available")
        except ImportError:
            print("   • PyTorch not installed")
    
    print()


# Flag to control whether to print startup info
_PRINT_STARTUP = True

def _suppress_startup():
    """Suppress startup information printing."""
    global _PRINT_STARTUP
    _PRINT_STARTUP = False

def _enable_startup():
    """Enable startup information printing."""
    global _PRINT_STARTUP
    _PRINT_STARTUP = True

