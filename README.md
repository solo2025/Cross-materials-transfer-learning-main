# 跨材料迁移学习项目 (Cross-Materials Transfer Learning)

## 项目概述

本项目实现了一个跨材料迁移学习系统，用于预测材料蠕变断裂寿命。主要研究从超级合金（Superalloys）到钛合金（Titanium Alloys）的机器学习模型迁移，通过CNN深度学习和传统机器学习算法的融合来提升预测性能。

## 项目结构

```
Cross-materials-transfer-learning-main/
├── Source CNN.py              # 源域CNN模型训练脚本
├── TrCNN.py                   # 迁移学习CNN脚本
├── Data fusion-based TL.py    # 数据融合迁移学习脚本
├── data.S1.csv               # 超级合金数据集 (754样本)
├── data.S2.csv               # 钛合金数据集 (89样本)
├── cnn.h5                    # 源域预训练CNN模型
├── cnn2.h5                   # 迁移学习后的CNN模型
└── README.md                 # 项目说明文档
```

## 数据集说明

### 超级合金数据集 (data.S1.csv)
- **样本数量**: 754个
- **特征数量**: 34个
- **主要特征**:
  - 热处理参数：固溶处理时间/温度、时效处理时间/温度
  - 化学成分：C, Si, Mn, P, S, Ni, Cr, Mo, Cu, W, Co, Al, N, Nb+Ta, B, V, Ti, Fe, Zr, Re, Y, Hf等
  - 测试条件：测试温度、测试应力
- **目标变量**: 蠕变断裂寿命 (Creep rupture life, 小时)

### 钛合金数据集 (data.S2.csv)
- **样本数量**: 89个
- **特征数量**: 24个
- **主要特征**:
  - 热处理参数：固溶处理、时效处理参数
  - 化学成分：C, Si, Ni, Mo, W, Al, N, Nb+Ta, B, V, Ti, Fe, Zr, Sn等
  - 测试条件：测试温度、测试应力
- **目标变量**: 蠕变断裂寿命 (Creep rupture life, 小时)

## 运行环境要求

### Python版本
- Python 3.7+

### 依赖包
```bash
pip install numpy pandas matplotlib scikit-learn keras tensorflow xgboost shap
```

### 主要依赖包列表
- `numpy`: 数值计算
- `pandas`: 数据处理
- `matplotlib`: 数据可视化
- `scikit-learn`: 机器学习算法
- `keras`: 深度学习框架
- `tensorflow`: 深度学习后端
- `xgboost`: 梯度提升算法
- `shap`: 模型可解释性分析

## 使用指南

### 1. 源域CNN模型训练

运行 `Source CNN.py` 在超级合金数据上训练基础CNN模型：

```bash
python "Source CNN.py"
```

**功能说明**:
- 读取 `Superalloys.csv` 数据
- 将34维特征重塑为6×6×1的图像格式
- 构建CNN网络：2个卷积层 + 全连接层
- 训练1500个epoch，使用Adam优化器
- 保存最佳模型为 `cnn.h5`
- 生成训练结果可视化图表

**输出文件**:
- `cnn.h5`: 预训练的CNN模型
- `Superalloys_CNN_lgRT.png`: 预测结果可视化图

### 2. 迁移学习

运行 `TrCNN.py` 进行跨材料迁移学习：

```bash
python TrCNN.py
```

**功能说明**:
- 加载预训练的 `cnn.h5` 模型
- 冻结卷积层权重，只训练全连接层
- 在钛合金数据上进行微调
- 保存迁移后的模型为 `cnn2.h5`
- 评估迁移学习效果

**配置参数**:
```python
train_input1 = "RT<=100.csv"           # 训练数据
test_input = "RT>100h, 35 points.csv"  # 测试数据
```

**输出文件**:
- `cnn2.h5`: 迁移学习后的模型
- `Trans_CNN>100h, 35 points.png`: 迁移学习结果图

### 3. 数据融合迁移学习

运行 `Data fusion-based TL.py` 比较多种机器学习算法：

```bash
python "Data fusion-based TL.py"
```

**功能说明**:
- 实现5种机器学习算法：随机森林、高斯过程回归、SVR、XGBoost、AdaBoost
- 使用网格搜索进行超参数优化
- 生成SHAP可解释性分析图
- 输出详细的性能评估指标

**算法配置**:

#### 随机森林 (Random Forest)
```python
parameters = {
    'n_estimators': [10, 20, 30],
    'max_depth': [15, 16, 17],
    'min_samples_split': [4, 5, 6],
    'min_samples_leaf': [3, 4, 5]
}
```

#### 高斯过程回归 (Gaussian Process)
```python
parameters = {
    'normalize_y': [False],
    'kernel': [None, DotProduct(), DotProduct() + WhiteKernel(), WhiteKernel()],
    'alpha': np.arange(0.001, 0.1, 0.001)
}
```

#### XGBoost
```python
parameters = {
    'n_estimators': [160, 170],
    'max_depth': [5, 6, 7],
    'min_child_weight': [2, 3, 4],
    'gamma': [0.001, 0.01, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_lambda': [2, 3, 5, 8],
    'reg_alpha': [0, 0.1]
}
```

**输出文件**:
- `RF_.png`: 随机森林结果图
- `GPR_.png`: 高斯过程回归结果图
- `SVR_.png`: 支持向量回归结果图
- `XGBoost_.png`: XGBoost结果图
- `AdaBoost_.png`: AdaBoost结果图
- SHAP可解释性分析图

## 评估指标

项目使用以下指标评估模型性能：

- **MSE (Mean Squared Error)**: 均方误差
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **RMSE (Root Mean Squared Error)**: 均方根误差
- **R² (R-squared)**: 决定系数
- **MAPE (Mean Absolute Percentage Error)**: 平均绝对百分比误差

## 运行顺序建议

1. **准备数据**: 确保CSV数据文件在正确位置
2. **训练源域模型**: 运行 `Source CNN.py`
3. **执行迁移学习**: 运行 `TrCNN.py`
4. **比较多种算法**: 运行 `Data fusion-based TL.py`

## 技术特点

### 迁移学习策略
- **特征提取**: 使用预训练的CNN提取材料特征
- **微调策略**: 冻结卷积层，只训练全连接层
- **数据融合**: 结合多个数据源提升模型泛化能力

### 数据预处理
- **标准化**: 使用MinMaxScaler进行特征标准化
- **特征重塑**: 将1D特征转换为2D图像格式 (6×6×1)
- **数据分割**: 自动分割训练集和测试集

### 模型可解释性
- **SHAP分析**: 提供特征重要性分析
- **可视化**: 自动生成预测vs真实值对比图
- **性能报告**: 详细的模型性能评估

## 注意事项

1. **文件路径**: 确保所有CSV文件路径正确
2. **内存使用**: 大数据集可能需要较多内存
3. **训练时间**: CNN训练可能需要较长时间
4. **依赖版本**: 注意Keras/TensorFlow版本兼容性

## 扩展功能

### 自定义配置
可以通过修改脚本中的配置部分来调整：
- 数据文件路径
- 模型超参数
- 训练epoch数
- 评估指标

### 添加新算法
在 `Data fusion-based TL.py` 中可以轻松添加新的机器学习算法进行对比。

## 联系信息

如有问题或建议，请通过项目仓库提交Issue。

## 许可证

本项目采用MIT许可证。
