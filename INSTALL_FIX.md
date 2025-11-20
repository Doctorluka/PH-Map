# 安装问题修复说明

## 问题描述

安装包后导入时出现错误：
```
ModuleNotFoundError: No module named 'phmap.core'
```

## 问题原因

`setup.py` 中的包配置不正确，导致子模块（`core`, `pl`, `models`, `utils`）没有被正确包含在安装包中。

## 修复方案

修改了 `setup.py`，使用 `find_packages()` 自动发现所有子包，并正确配置包结构：

```python
# 修复前（错误）
packages=['phmap', 'phmap.core', 'phmap.pl', 'phmap.models', 'phmap.utils'],
package_dir={
    'phmap': '.',
    'phmap.core': 'core',  # 这个配置方式不正确
    ...
}

# 修复后（正确）
subpackages = find_packages(where='.')
packages = ['phmap'] + [f'phmap.{pkg}' for pkg in subpackages]
package_dir={
    'phmap': '.',  # 当前目录就是phmap包根目录
}
```

## 验证结果

修复后，所有功能正常：

```python
import phmap  # ✓ 成功
import phmap.core  # ✓ 成功
import phmap.pl  # ✓ 成功
import phmap.models  # ✓ 成功
```

## 安装方法

```bash
# 1. 激活环境
mamba activate phmap-env

# 2. 进入phmap目录
cd /home/data/fhz/project/phmap_package/phmap

# 3. 重新安装（如果之前安装过）
pip uninstall phmap -y
pip install -e .
```

## 依赖说明

`setup.py` 中已完整配置所有依赖（`install_requires`），安装时会自动安装：

- torch>=2.0.0
- scanpy>=1.9.0
- pandas>=1.5.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- anndata>=0.9.0
- scipy>=1.9.0

## 测试

安装后测试：

```python
import phmap
# 应该看到启动信息（logo、版本、GPU信息）

# 测试所有功能
phmap.list_available_models()
phmap.pl  # 绘图模块
```

## 状态

✅ **问题已修复，包可以正常安装和导入**

