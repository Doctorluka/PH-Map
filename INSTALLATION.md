# PH-Map 安装指南

## 快速安装

### 方法1：开发模式安装（推荐）

```bash
# 1. 激活phmap-env环境
mamba activate phmap-env

# 2. 进入phmap目录
cd /home/data/fhz/project/phmap_package/phmap

# 3. 安装包（开发模式，修改代码后无需重新安装）
pip install -e .
```

### 方法2：普通安装

```bash
# 1. 激活phmap-env环境
mamba activate phmap-env

# 2. 进入phmap目录
cd /home/data/fhz/project/phmap_package/phmap

# 3. 安装包
pip install .
```

## 依赖说明

### setup.py中的依赖配置

`setup.py` 中已经配置了所有必需的依赖（`install_requires`），安装时会自动安装：

```python
install_requires=[
    'torch>=2.0.0',           # PyTorch深度学习框架
    'scanpy>=1.9.0',          # 单细胞分析
    'pandas>=1.5.0',          # 数据处理
    'numpy>=1.24.0',          # 数值计算
    'scikit-learn>=1.3.0',    # 机器学习工具
    'matplotlib>=3.7.0',      # 绘图
    'seaborn>=0.12.0',        # 统计绘图
    'anndata>=0.9.0',         # AnnData数据结构
    'scipy>=1.9.0',           # 科学计算
]
```

### 依赖安装方式

**方式1：自动安装（推荐）**
```bash
# 直接安装phmap，依赖会自动安装
pip install -e .
```

**方式2：手动安装依赖**
```bash
# 如果需要先安装依赖
pip install -r requirements.txt

# 然后安装phmap
pip install -e .
```

**方式3：使用phmap-env环境（推荐）**
```bash
# phmap-env环境已经包含了所有依赖
# 只需要安装phmap包本身
mamba activate phmap-env
cd /home/data/fhz/project/phmap_package/phmap
pip install -e .
```

## 验证安装

安装完成后，验证是否成功：

```python
# 测试导入
import phmap

# 应该看到启动信息：
# - PHMAP ASCII艺术图标
# - 版本号
# - GPU检测信息

# 检查版本
print(f"PH-Map version: {phmap.__version__}")

# 检查可用模型
models = phmap.list_available_models()
print("Available models:", list(models.keys()))

# 检查API
print("Available functions:", [x for x in dir(phmap) if not x.startswith('_')])
```

## 完整安装步骤（使用phmap-env）

```bash
# 1. 激活环境
mamba activate phmap-env

# 2. 进入phmap目录
cd /home/data/fhz/project/phmap_package/phmap

# 3. 安装包（开发模式）
pip install -e .

# 4. 验证安装
python -c "import phmap; print('✓ Installation successful!')"
```

## 依赖版本说明

### 最小版本要求（setup.py中）

- Python >= 3.11
- torch >= 2.0.0
- scanpy >= 1.9.0
- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- anndata >= 0.9.0
- scipy >= 1.9.0

### phmap-env环境中的实际版本

如果您使用phmap-env环境，实际安装的版本可能更高（见`docs/requirements_phmap_env.txt`）：
- torch==2.5.1+cu121
- scanpy==1.11.5
- pandas==1.5.3
- numpy==1.26.4
- 等等...

这些更高版本都是兼容的，可以正常使用。

## 常见问题

### Q1: 安装时提示缺少依赖？

**A:** setup.py中已经配置了依赖，但如果您想手动安装：
```bash
pip install torch scanpy pandas numpy scikit-learn matplotlib seaborn anndata scipy
```

### Q2: 如何更新依赖？

**A:** 
```bash
# 更新所有依赖到最新兼容版本
pip install --upgrade torch scanpy pandas numpy scikit-learn matplotlib seaborn anndata scipy

# 或者重新安装phmap（会自动检查依赖）
pip install -e . --upgrade
```

### Q3: 安装后导入失败？

**A:** 检查以下几点：
1. 确保在正确的环境中：`which python` 应该指向phmap-env
2. 确保包已安装：`pip list | grep phmap`
3. 检查Python版本：`python --version` 应该 >= 3.11

### Q4: 模型文件未找到？

**A:** 确保安装时包含了模型文件：
```bash
# 检查模型文件是否存在
ls phmap/models/full_model/

# 如果不存在，重新安装
pip install -e . --force-reinstall
```

## 卸载

如果需要卸载phmap：

```bash
pip uninstall phmap
```

## 重新安装

如果需要重新安装：

```bash
# 卸载旧版本
pip uninstall phmap -y

# 重新安装
cd /home/data/fhz/project/phmap_package/phmap
pip install -e .
```

