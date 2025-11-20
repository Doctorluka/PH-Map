# PH-Map 安装和使用指南

## 安装

### 开发模式安装（推荐）

由于 `phmap` 是一个完全独立的包，所有配置文件都在 `phmap/` 目录内。

```bash
cd /home/data/fhz/project/phmap_package/phmap
pip install -e .
```

### 普通安装

```bash
cd /home/data/fhz/project/phmap_package/phmap
pip install .
```

## 验证安装

```python
import phmap
print(f"PH-Map version: {phmap.__version__}")

# 检查可用模型
models = phmap.list_available_models()
print("Available models:", list(models.keys()))

# 检查API
print("Available functions:", [x for x in dir(phmap) if not x.startswith('_')])
```

## 快速使用示例

```python
import phmap
import scanpy as sc

# 加载查询数据
adata = sc.read_h5ad('query_data.h5ad')

# 使用默认模型进行预测（最简单的方式）
result = phmap.predict(adata, return_probabilities=True)

# 可视化结果
phmap.pl.plot_probability_bar(result, label_columns=['anno_lv4'])

# 将预测结果添加到AnnData
adata = result.to_adata(adata)
```

## 包结构

```
phmap/                          # 完全独立的Python包目录
├── setup.py                    # 安装配置文件
├── pyproject.toml              # 现代Python包配置
├── MANIFEST.in                 # 包含非Python文件
├── LICENSE                     # 许可证
├── README.md                   # 使用说明
├── requirements.txt            # 依赖列表
├── __init__.py                 # 主入口
├── version.py                  # 版本信息
├── core/                       # 核心功能模块
│   ├── __init__.py
│   ├── classifier.py
│   └── predictor.py
├── pl/                         # Plotting模块
│   ├── __init__.py
│   └── plotting.py
└── models/                     # 预训练模型
    ├── __init__.py
    ├── __model_registry__.py
    └── full_model/
        ├── cell_type_classifier_full.pth
        └── cell_type_classifier_full_metadata.pkl
```

## API概览

### 核心功能

- `phmap.build_model()` - 训练新模型
- `phmap.predict()` - 预测（默认使用full_model）
- `phmap.evaluate()` - 评估预测结果
- `phmap.load_pretrained_model()` - 加载预训练模型
- `phmap.list_available_models()` - 列出可用模型

### 可视化功能

- `phmap.pl.plot_probability_heatmap()` - 概率热图
- `phmap.pl.plot_probability_bar()` - 概率柱状图
- `phmap.pl.plot_cell_type_counts()` - 细胞类型计数
- `phmap.pl.plot_probability_distribution()` - 概率分布

## 注意事项

1. **安装位置**：必须在 `phmap/` 目录内运行 `pip install -e .`
2. **完全隔离**：`phmap/` 目录是一个完全独立的Python包，所有配置文件都在此目录内
3. **确保已安装所有依赖**（见requirements.txt或setup.py中的install_requires）
4. **首次使用**：首次使用`phmap.predict()`时会自动加载full_model（可能需要一些时间）
5. **模型文件较大**（约17MB），确保有足够的磁盘空间
6. **建议使用GPU加速**（如果可用）

## 故障排除

如果遇到导入错误：
1. 确保已正确安装包：`cd phmap && pip install -e .`
2. 检查Python版本：需要 >= 3.11
3. 检查依赖是否安装：`pip list | grep torch`

如果模型加载失败：
1. 检查模型文件是否存在：`ls phmap/models/full_model/`
2. 确保包正确安装，模型文件被包含在包中
