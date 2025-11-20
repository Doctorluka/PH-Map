# PH-Map 启动信息功能

## 功能说明

当用户导入 `phmap` 时，会自动显示启动信息，包括：
1. **ASCII艺术图标** - PH-Map的logo
2. **版本信息** - 当前包的版本号
3. **GPU/CUDA检测** - 自动检测并显示GPU信息

## 显示效果

当用户运行 `import phmap` 时，会看到类似以下输出：

```
   ____  _   __  __                  
  |  _ \| | |  \/  | __ _ _ __ ___   
  | |_) | |_| |\/| |/ _` | '_ ` _ \  
  |  __/|  _  | |  | (_| | | | | | | 
  |_|   |_| |_|_|  |_|\__,_|_| |_| |_|
                                      
  🔬 Multi-Task Cell Type Classification
    
🔖 Version: 0.1.0

🧬 Detecting CUDA devices…
✅ [GPU 0] NVIDIA GeForce RTX 4090
   • Total memory: 23.6 GB
   • Compute capability: 8.9
   • CUDA version: 12.1
```

如果没有GPU，会显示：

```
🧬 Detecting CUDA devices…
⚠️  No GPU detected, using CPU
   • PyTorch CUDA support: Not available
```

## 实现细节

### 文件结构

- `phmap/_startup.py` - 启动信息模块
  - `_get_gpu_info()` - GPU检测函数
  - `_print_startup_info()` - 打印启动信息
  - `_suppress_startup()` - 禁用启动信息（高级用户）
  - `_enable_startup()` - 启用启动信息

- `phmap/__init__.py` - 在导入时自动调用启动信息

### GPU检测功能

`_get_gpu_info()` 函数会：
1. 尝试导入 `torch`
2. 检查 `torch.cuda.is_available()`
3. 如果可用，获取以下信息：
   - GPU设备数量
   - 当前使用的GPU
   - GPU名称
   - 总内存（GB）
   - 计算能力
   - CUDA版本

### 控制启动信息

如果需要禁用启动信息（例如在脚本中批量处理时），可以使用：

```python
import phmap
phmap._suppress_startup()  # 禁用启动信息

# 重新导入不会显示启动信息
import phmap  # 不会显示启动信息
```

或者重新启用：

```python
phmap._enable_startup()  # 重新启用启动信息
```

**注意**：由于Python的导入机制，`_suppress_startup()` 需要在第一次导入之前调用，或者需要重新加载模块。

## 自定义

### 修改ASCII艺术

编辑 `phmap/_startup.py` 中的 `logo` 变量：

```python
logo = """
   ____  _   __  __                  
  |  _ \| | |  \/  | __ _ _ __ ___   
  ...
"""
```

### 修改显示内容

编辑 `_print_startup_info()` 函数来调整显示的信息和格式。

## 测试

测试启动信息显示：

```python
# 在phmap-env环境中
import phmap
```

应该能看到启动信息输出。

## 注意事项

1. 启动信息只在第一次导入时显示
2. 如果PyTorch未安装，会显示相应的提示
3. GPU检测依赖于PyTorch的CUDA支持
4. 启动信息不会影响包的正常功能

