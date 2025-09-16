# 合金材料机器学习流水线

这是一个用于合金材料性能预测和设计的机器学习流水线，整合了数据分析、模型训练、合金设计等功能。

## 项目结构

```
transfer/
├── core/                          # 核心模块
│   ├── models.py                  # 神经网络模型定义
│   ├── data_processing.py         # 数据处理和特征工程
│   ├── training.py               # 模型训练和评估
│   ├── visualization.py          # 可视化和分析
│   └── alloy_design.py           # 合金设计和优化
├── data/                         # 数据文件
│   ├── VAM_LWRHEAS_property_*.csv
│   └── AM_LWRHEAS_property_*.csv
├── main_pipeline.py              # 主流程脚本
├── config.json                   # 配置文件
├── parameter_cal.py              # 参数计算模块
└── README.md                     # 项目说明
```

## 功能特性

### 1. 数据分析
- 成分分布分析
- 相关性矩阵计算
- 数据质量检查
- 统计报告生成

### 2. 模型训练
- VAM数据神经网络训练
- AM数据迁移学习
- 交叉验证
- 模型评估和可视化

### 3. 合金设计
- 随机合金生成
- 遗传算法优化
- Pareto前沿分析
- 性能预测

### 4. 模型解释
- SHAP值分析
- 特征重要性
- t-SNE可视化
- 模型解释报告

## 安装依赖

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn plotly shap deap joblib
```

## 使用方法

### 1. 运行完整流水线

```bash
python main_pipeline.py --mode full
```

### 2. 只进行数据分析

```bash
python main_pipeline.py --mode analyze --data_type vam
```

### 3. 只训练模型

```bash
python main_pipeline.py --mode train --data_type vam
```

### 4. 只进行合金设计

```bash
python main_pipeline.py --mode design
```

## 配置说明

配置文件 `config.json` 包含以下主要设置：

- `data_paths`: 数据文件路径
- `feature_columns`: 特征列名定义
- `model_config`: 模型配置参数
- `output_dirs`: 输出目录设置
- `composition_constraints`: 成分约束条件
- `process_parameters`: 工艺参数设置
- `optimization`: 优化算法参数
- `visualization`: 可视化设置
- `logging`: 日志配置

### 配置管理器

项目包含一个强大的配置管理器 (`core/config_manager.py`)，提供：

- 配置文件加载和验证
- 动态配置更新
- 实验配置管理
- 数据路径验证
- 日志配置

### 配置测试

运行配置测试：
```bash
python test_config.py
```

## 核心模块说明

### models.py
- `VAMNet`: VAM数据预测模型
- `AMFeatureNet`: AM特征提取网络
- `TransferLearningNet`: 迁移学习组合网络
- `ModelManager`: 模型管理器

### data_processing.py
- `DataProcessor`: 数据处理器
- `FeatureSelector`: 特征选择器
- `DataAnalyzer`: 数据分析器

### config_manager.py
- `ConfigManager`: 配置管理器
- 配置文件加载和验证
- 动态配置更新
- 实验配置管理

### training.py
- `ModelTrainer`: 模型训练器
- `ModelEvaluator`: 模型评估器

### visualization.py
- `DataVisualizer`: 数据可视化器
- `ModelExplainer`: 模型解释器
- `ReportGenerator`: 报告生成器

### alloy_design.py
- `AlloyGenerator`: 合金生成器
- `GeneticOptimizer`: 遗传算法优化器
- `AlloyDesigner`: 合金设计器

## 输出结果

运行流水线后，会在以下目录生成结果：

- `models/`: 训练好的模型文件
- `results/`: 分析结果和预测数据
- `plots/`: 可视化图表
- `reports/`: 分析报告

## 优化改进

相比原始脚本，新版本具有以下优势：

1. **模块化设计**: 功能分离，易于维护和扩展
2. **统一接口**: 提供一致的API接口
3. **配置管理**: 通过配置文件管理参数
4. **错误处理**: 完善的异常处理机制
5. **代码复用**: 消除重复代码
6. **文档完善**: 详细的文档和注释
7. **可扩展性**: 易于添加新功能

## 注意事项

1. 确保数据文件路径正确
2. 根据硬件配置调整模型参数
3. 大规模优化时注意计算资源
4. 定期保存中间结果

## 许可证

本项目采用MIT许可证。
