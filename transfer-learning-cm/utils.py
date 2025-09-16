"""
通用工具模块 - 数据预处理、模型评估、可视化等功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

from config import PLOT_CONFIG, EVALUATION_CONFIG


class DataProcessor:
    """数据预处理类"""
    
    def __init__(self, scaler_type='minmax'):
        """
        初始化数据处理器
        
        Args:
            scaler_type (str): 缩放器类型，'minmax' 或 'standard'
        """
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
    
    def reshape_for_cnn(self, data, target_shape=(6, 6, 1)):
        """
        将数据重塑为CNN输入格式
        
        Args:
            data (np.array): 输入数据
            target_shape (tuple): 目标形状
            
        Returns:
            np.array: 重塑后的数据
        """
        reshaped_data = []
        for i in range(len(data)):
            sample = data[i, :]
            # 填充到36个特征 (6x6)
            padded_sample = np.pad(sample, (0, 2), 'constant', constant_values=(0, 0))
            # 重塑为6x6x1
            reshaped_sample = padded_sample.reshape(*target_shape)
            reshaped_data.append(reshaped_sample)
        
        return np.array(reshaped_data)
    
    def prepare_cnn_data(self, X, y, test_size=0.2, random_state=42):
        """
        准备CNN训练数据
        
        Args:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标数据
            test_size (float): 测试集比例
            random_state (int): 随机种子
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 重塑为CNN格式
        X_reshaped = self.reshape_for_cnn(X_scaled)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test


class ModelEvaluator:
    """模型评估类"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        计算评估指标
        
        Args:
            y_true (array-like): 真实值
            y_pred (array-like): 预测值
            
        Returns:
            dict: 包含各种评估指标的字典
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    @staticmethod
    def print_metrics(metrics, model_name="模型"):
        """
        打印评估指标
        
        Args:
            metrics (dict): 评估指标字典
            model_name (str): 模型名称
        """
        print(f"{model_name}评估结果:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.4f}")
        print()


class Visualizer:
    """可视化类"""
    
    def __init__(self):
        """初始化可视化器"""
        self.config = PLOT_CONFIG
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """设置matplotlib参数"""
        plt.rcParams['font.sans-serif'] = [self.config['font_family']]
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_predictions(self, y_train_pred, y_train_true, y_test_pred, y_test_true, 
                        title="预测结果", save_path=None, train_labels=None):
        """
        绘制预测结果散点图
        
        Args:
            y_train_pred (array-like): 训练集预测值
            y_train_true (array-like): 训练集真实值
            y_test_pred (array-like): 测试集预测值
            y_test_true (array-like): 测试集真实值
            title (str): 图表标题
            save_path (str): 保存路径
            train_labels (list): 训练集标签列表
        """
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        # 绘制训练集
        if train_labels is None:
            ax.plot(y_train_pred, y_train_true, self.config['markers']['train1'], 
                   color=self.config['colors']['train1'], label='训练集')
        else:
            # 如果有多个训练集标签
            for i, (pred, true, label) in enumerate(zip(y_train_pred, y_train_true, train_labels)):
                ax.plot(pred, true, self.config['markers'][f'train{i+1}'], 
                       color=self.config['colors'][f'train{i+1}'], label=label)
        
        # 绘制测试集
        ax.plot(y_test_pred, y_test_true, self.config['markers']['test'], 
               color=self.config['colors']['test'], label='测试集')
        
        # 绘制对角线
        min_val = min(min(y_train_true), min(y_test_true), min(y_train_pred), min(y_test_pred))
        max_val = max(max(y_train_true), max(y_test_true), max(y_train_pred), max(y_test_pred))
        ax.plot([min_val, max_val], [min_val, max_val], '--', color=self.config['colors']['line'])
        
        # 设置坐标轴范围
        ax.set_xlim((min_val, max_val))
        ax.set_ylim((min_val, max_val))
        
        # 设置标签和标题
        ax.set_xlabel('预测值', weight='bold', fontproperties=self.config['font_family'], 
                     fontsize=self.config['font_size'])
        ax.set_ylabel('真实值', weight='bold', fontproperties=self.config['font_family'], 
                     fontsize=self.config['font_size'])
        ax.set_title(title, fontsize=self.config['font_size'], 
                    fontproperties=self.config['font_family'], fontweight='bold')
        
        # 设置刻度
        ax.tick_params(labelsize=self.config['tick_size'])
        plt.yticks(fontproperties=self.config['font_family'], size=self.config['tick_size'])
        plt.xticks(fontproperties=self.config['font_family'], size=self.config['tick_size'])
        
        # 添加图例
        ax.legend(loc='best')
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        plt.show()
    
    def plot_data_fusion_predictions(self, y_train_pred1, y_train_true1, y_train_pred2, y_train_true2,
                                   y_test_pred, y_test_true, model_name, save_path=None):
        """
        绘制数据融合模型的预测结果
        
        Args:
            y_train_pred1 (array-like): 训练集1预测值
            y_train_true1 (array-like): 训练集1真实值
            y_train_pred2 (array-like): 训练集2预测值
            y_train_true2 (array-like): 训练集2真实值
            y_test_pred (array-like): 测试集预测值
            y_test_true (array-like): 测试集真实值
            model_name (str): 模型名称
            save_path (str): 保存路径
        """
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        # 绘制训练集1
        ax.plot(y_train_pred1, y_train_true1, self.config['markers']['train1'], 
               color=self.config['colors']['train1'], label='训练集1')
        
        # 绘制训练集2
        ax.plot(y_train_pred2, y_train_true2, self.config['markers']['train2'], 
               color=self.config['colors']['train2'], label='训练集2')
        
        # 绘制测试集
        ax.plot(y_test_pred, y_test_true, self.config['markers']['test'], 
               color=self.config['colors']['test'], label='测试集')
        
        # 绘制对角线
        ax.plot([0, 6], [0, 6], '--', color=self.config['colors']['line'])
        ax.set_xlim((0, 6))
        ax.set_ylim((0, 6))
        
        # 设置标签和标题
        ax.set_xlabel('预测值', weight='bold', fontproperties=self.config['font_family'], 
                     fontsize=self.config['font_size'])
        ax.set_ylabel('真实值', weight='bold', fontproperties=self.config['font_family'], 
                     fontsize=self.config['font_size'])
        ax.set_title(f'{model_name}_预测结果', fontsize=self.config['font_size'], 
                    fontproperties=self.config['font_family'], fontweight='bold')
        
        # 设置刻度
        ax.tick_params(labelsize=self.config['tick_size'])
        plt.yticks(fontproperties=self.config['font_family'], size=self.config['tick_size'])
        plt.xticks(fontproperties=self.config['font_family'], size=self.config['tick_size'])
        
        # 添加图例
        ax.legend(loc='best')
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        plt.show()


def load_and_prepare_data(file_path, target_column, feature_columns=None, drop_columns=None):
    """
    加载和准备数据
    
    Args:
        file_path (str): 数据文件路径
        target_column (str): 目标列名
        feature_columns (list): 特征列名列表，None表示使用除目标列外的所有列
        drop_columns (list): 需要删除的列名列表
        
    Returns:
        tuple: (X, y) 特征矩阵和目标向量
    """
    # 读取数据
    data = pd.read_csv(file_path)
    
    # 删除指定列
    if drop_columns:
        data = data.drop(drop_columns, axis=1)
    
    # 分离特征和目标
    if feature_columns is None:
        X = data.drop(target_column, axis=1)
    else:
        X = data[feature_columns]
    
    y = data[target_column]
    
    return X, y
