"""
迁移学习CNN模型训练脚本
使用预训练的源CNN模型进行迁移学习
"""

import numpy as np
import pandas as pd
from keras import layers, models
from keras.models import load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from config import DATA_CONFIG, MODEL_CONFIG
from utils import DataProcessor, ModelEvaluator, Visualizer, load_and_prepare_data


class TransferCNN:
    """迁移学习CNN模型类"""
    
    def __init__(self, source_model_path='cnn.h5', config=None):
        """
        初始化迁移学习CNN模型
        
        Args:
            source_model_path (str): 源模型路径
            config (dict): 模型配置，如果为None则使用默认配置
        """
        self.source_model_path = source_model_path
        self.config = config or MODEL_CONFIG['cnn']
        self.data_processor = DataProcessor()
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        self.model = None
        self.history = None
        self.base_model = None
    
    def load_base_model(self):
        """
        加载预训练的基模型
        
        Returns:
            keras.Model: 预训练的基模型
        """
        print(f"加载预训练模型: {self.source_model_path}")
        base_model = load_model(self.source_model_path)
        
        # 移除最后一层（输出层）
        base_model.pop()
        
        # 冻结基模型参数
        base_model.trainable = False
        
        print("基模型层结构:")
        for i, layer in enumerate(base_model.layers):
            print(f"  层 {i}: {layer.name} - 可训练: {layer.trainable}")
        
        self.base_model = base_model
        return base_model
    
    def build_transfer_model(self):
        """构建迁移学习模型"""
        if self.base_model is None:
            self.load_base_model()
        
        model = models.Sequential()
        
        # 添加预训练的基模型
        model.add(self.base_model)
        
        # 添加新的全连接层
        for units in self.config['dense_units']:
            model.add(layers.Dense(units, activation=self.config['activation']))
        
        # 添加新的输出层
        model.add(layers.Dense(self.config['output_units']))
        
        return model
    
    def compile_model(self, model):
        """
        编译模型
        
        Args:
            model: Keras模型对象
        """
        optimizer = optimizers.Adam(lr=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss=self.config['loss'])
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, model_save_path='cnn2.h5'):
        """
        训练迁移学习模型
        
        Args:
            X_train (np.array): 训练特征
            y_train (np.array): 训练标签
            X_test (np.array): 测试特征
            y_test (np.array): 测试标签
            model_save_path (str): 模型保存路径
            
        Returns:
            keras.Model: 训练好的模型
        """
        # 构建和编译模型
        self.model = self.build_transfer_model()
        self.model = self.compile_model(self.model)
        
        # 打印模型结构
        print("迁移学习模型结构:")
        self.model.summary()
        print()
        
        # 设置回调函数
        checkpoint = ModelCheckpoint(
            model_save_path, 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            mode='min'
        )
        
        # 训练模型
        print("开始训练迁移学习模型...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[checkpoint],
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        print("迁移学习模型训练完成!")
        return self.model
    
    def evaluate_model(self, X_test, y_test, model_path=None):
        """
        评估模型性能
        
        Args:
            X_test (np.array): 测试特征
            y_test (np.array): 测试标签
            model_path (str): 模型路径，如果为None则使用当前模型
            
        Returns:
            dict: 评估指标
        """
        # 加载最佳模型
        if model_path:
            model = load_model(model_path)
        else:
            model = self.model
        
        # 进行预测
        y_pred = model.predict(X_test).flatten()
        
        # 计算评估指标
        metrics = self.evaluator.calculate_metrics(y_test, y_pred)
        
        # 打印评估结果
        self.evaluator.print_metrics(metrics, "迁移学习CNN模型")
        
        return metrics, y_pred
    
    def visualize_results(self, X_train, y_train, X_test, y_test, model_path=None, 
                         save_path='Trans_CNN_RT>100h_35_points.png'):
        """
        可视化预测结果
        
        Args:
            X_train (np.array): 训练特征
            y_train (np.array): 训练标签
            X_test (np.array): 测试特征
            y_test (np.array): 测试标签
            model_path (str): 模型路径
            save_path (str): 图片保存路径
        """
        # 加载最佳模型
        if model_path:
            model = load_model(model_path)
        else:
            model = self.model
        
        # 进行预测
        y_train_pred = model.predict(X_train).flatten()
        y_test_pred = model.predict(X_test).flatten()
        
        # 可视化结果
        self.visualizer.plot_predictions(
            y_train_pred, y_train,
            y_test_pred, y_test,
            title='Trans_CNN_RT>100h, 35 points',
            save_path=save_path
        )
        
        return y_train_pred, y_test_pred


def prepare_transfer_data(train_file, test_file, target_column):
    """
    准备迁移学习数据
    
    Args:
        train_file (str): 训练数据文件路径
        test_file (str): 测试数据文件路径
        target_column (str): 目标列名
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # 读取训练数据
    df_train = pd.read_csv(train_file)
    df_train = df_train.iloc[:, 1:]  # 删除第一列
    
    # 读取测试数据
    df_test = pd.read_csv(test_file)
    df_test = df_test.iloc[:, 1:]  # 删除第一列
    
    # 合并数据用于统一标准化
    df_all = pd.concat([df_train, df_test], axis=0)
    train_size = df_train.shape[0]
    
    # 分离特征和目标
    X_all = df_all.drop(target_column, axis=1)
    y_all = df_all[target_column]
    
    # 标准化特征
    data_processor = DataProcessor()
    X_all_scaled = data_processor.scaler.fit_transform(X_all)
    
    # 重塑为CNN格式
    X_all_reshaped = data_processor.reshape_for_cnn(X_all_scaled)
    
    # 分离训练和测试数据
    X_train = X_all_reshaped[0:train_size]
    X_test = X_all_reshaped[train_size:]
    y_train = y_all[0:train_size]
    y_test = y_all[train_size:]
    
    return X_train, X_test, y_train, y_test


def main():
    """主函数"""
    print("=" * 50)
    print("迁移学习CNN模型训练")
    print("=" * 50)
    
    # 获取配置
    data_config = DATA_CONFIG['transfer_cnn']
    
    # 准备数据
    print("准备迁移学习数据...")
    X_train, X_test, y_train, y_test = prepare_transfer_data(
        train_file=data_config['train_file'],
        test_file=data_config['test_file'],
        target_column=data_config['target_column']
    )
    
    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
    print()
    
    # 创建迁移学习模型
    print("创建迁移学习CNN模型...")
    transfer_cnn = TransferCNN(source_model_path='cnn.h5')
    
    # 训练模型
    model = transfer_cnn.train_model(X_train, y_train, X_test, y_test)
    
    # 评估模型
    print("评估模型性能...")
    metrics, y_test_pred = transfer_cnn.evaluate_model(X_test, y_test, 'cnn2.h5')
    
    # 可视化结果
    print("生成可视化结果...")
    transfer_cnn.visualize_results(X_train, y_train, X_test, y_test, 'cnn2.h5')
    
    # 保存预测结果
    print("保存预测结果...")
    results_df = pd.DataFrame({
        'predicted': y_test_pred,
        'actual': y_test
    })
    # results_df.to_csv('transfer_cnn_predictions.csv', index=False)
    print("预测结果已保存到 transfer_cnn_predictions.csv")
    
    print("=" * 50)
    print("迁移学习CNN模型训练完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
