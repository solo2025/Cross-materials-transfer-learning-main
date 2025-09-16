# 跨材料迁移学习项目重构

## 项目概述

本项目实现了基于CNN和传统机器学习算法的跨材料迁移学习，用于材料性能预测。重构后的代码具有更好的结构、可读性和可维护性。

## 文件结构

```
├── config.py                    # 配置模块 - 统一管理所有配置参数
├── utils.py                     # 通用工具模块 - 数据预处理、模型评估、可视化等
├── source_cnn_refactored.py    # 源CNN模型训练脚本（重构版）
├── transfer_cnn_refactored.py  # 迁移学习CNN模型训练脚本（重构版）
├── data_fusion_refactored.py   # 数据融合迁移学习脚本（重构版）
├── Source CNN.py               # 原始源CNN脚本
├── TrCNN.py                    # 原始迁移学习CNN脚本
├── Data fusion-based TL.py     # 原始数据融合脚本
└── README.md                   # 项目说明文档
```

## 重构改进

### 1. 模块化设计
- **config.py**: 统一管理所有配置参数，包括数据路径、模型参数、可视化设置等
- **utils.py**: 提供通用的数据预处理、模型评估、可视化功能
- 每个主要脚本都采用面向对象的设计，提高代码复用性

### 2. 代码结构优化
- 将原本的脚本式代码重构为类和方法
- 提取公共功能到工具模块
- 统一错误处理和日志输出

### 3. 配置管理
- 所有硬编码的参数都移到配置文件中
- 支持不同实验的配置切换
- 便于参数调优和实验管理

### 4. 功能增强
- 添加了更详细的进度提示和结果输出
- 改进了可视化功能，支持多种图表类型
- 增强了模型评估功能，提供更全面的指标

## 使用方法

### 1. 源CNN模型训练

```bash
python source_cnn_refactored.py
```

**功能**:
- 训练基础CNN模型
- 生成预测结果可视化
- 保存训练好的模型

### 2. 迁移学习CNN模型训练

```bash
python transfer_cnn_refactored.py
```

**功能**:
- 使用预训练的源CNN模型进行迁移学习
- 冻结基础层，只训练新的分类层
- 适用于目标域数据较少的情况

### 3. 数据融合迁移学习

```bash
python data_fusion_refactored.py
```

**功能**:
- 使用多种机器学习算法进行数据融合
- 包括随机森林、高斯过程回归、SVR、XGBoost、AdaBoost
- 自动超参数优化和模型比较

## 配置说明

### 数据配置 (config.py)

```python
DATA_CONFIG = {
    'source_cnn': {
        'data_file': 'Superalloys.csv',
        'target_column': 'class',
        'test_size': 0.2,
        'random_state': 42
    },
    # ... 其他配置
}
```

### 模型配置 (config.py)

```python
MODEL_CONFIG = {
    'cnn': {
        'input_shape': (6, 6, 1),
        'conv_filters': [8, 16],
        'epochs': 1500,
        'batch_size': 50,
        # ... 其他参数
    },
    # ... 其他模型配置
}
```

## 主要类和方法

### DataProcessor 类
- `reshape_for_cnn()`: 将数据重塑为CNN输入格式
- `prepare_cnn_data()`: 准备CNN训练数据

### ModelEvaluator 类
- `calculate_metrics()`: 计算评估指标
- `print_metrics()`: 打印评估结果

### Visualizer 类
- `plot_predictions()`: 绘制预测结果散点图
- `plot_data_fusion_predictions()`: 绘制数据融合模型结果

### SourceCNN 类
- `build_model()`: 构建CNN模型
- `train_model()`: 训练模型
- `evaluate_model()`: 评估模型性能

### TransferCNN 类
- `load_base_model()`: 加载预训练模型
- `build_transfer_model()`: 构建迁移学习模型

### MLModelTrainer 类
- `train_random_forest()`: 训练随机森林模型
- `train_gaussian_process()`: 训练高斯过程回归模型
- `train_svr()`: 训练支持向量回归模型
- `train_xgboost()`: 训练XGBoost模型
- `train_adaboost()`: 训练AdaBoost模型

## 依赖库

```
numpy
pandas
scikit-learn
keras/tensorflow
matplotlib
xgboost
shap
```

## 注意事项

1. 确保数据文件路径正确
2. 根据实际数据调整配置参数
3. 训练时间较长，建议在GPU环境下运行
4. 生成的图片和模型文件会保存在当前目录

## 重构优势

1. **可维护性**: 模块化设计便于维护和扩展
2. **可复用性**: 通用功能可在不同脚本中复用
3. **可配置性**: 所有参数集中管理，便于实验
4. **可读性**: 清晰的类和方法结构，代码更易理解
5. **可扩展性**: 易于添加新的模型和功能
