"""
源CNN模型训练脚本
用于训练基础CNN模型，作为迁移学习的基础模型
"""

import numpy as np
import pandas as pd
from keras import layers, models
from keras.models import load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from config import DATA_CONFIG, MODEL_CONFIG
from utils import DataProcessor, ModelEvaluator, Visualizer, load_and_prepare_data


class SourceCNN:
    """源CNN模型类"""
    
    def __init__(self, config=None):
        """
        初始化源CNN模型
        
        Args:
            config (dict): 模型配置，如果为None则使用默认配置
        """
        self.config = config or MODEL_CONFIG['cnn']
        self.data_processor = DataProcessor()
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        self.model = None
        self.history = None
    
    def build_model(self):
        """构建CNN模型"""
        model = models.Sequential()
        
        # 第一个卷积层
        model.add(layers.Conv2D(
            self.config['conv_filters'][0], 
            self.config['conv_kernel_size'], 
            activation=self.config['activation'],
            padding=self.config['padding'], 
            input_shape=self.config['input_shape']
        ))
        
        # 第二个卷积层
        model.add(layers.Conv2D(
            self.config['conv_filters'][1], 
            self.config['conv_kernel_size'], 
            padding=self.config['padding'],
            activation=self.config['activation']
        ))
        
        # 展平层
        model.add(layers.Flatten())
        
        # 全连接层
        for units in self.config['dense_units']:
            model.add(layers.Dense(units, activation=self.config['activation']))
        
        # 输出层
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
    
    def train_model(self, X_train, y_train, X_test, y_test, model_save_path='cnn.h5'):
        """
        训练模型
        
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
        self.model = self.build_model()
        self.model = self.compile_model(self.model)
        
        # 打印模型结构
        print("模型结构:")
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
        print("开始训练模型...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[checkpoint],
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        print("模型训练完成!")
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
        self.evaluator.print_metrics(metrics, "源CNN模型")
        
        return metrics, y_pred
    
    def visualize_results(self, X_train, y_train, X_test, y_test, model_path=None, 
                         save_path='Superalloys_CNN_lgRT.png'):
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
            title='Superalloys_CNN',
            save_path=save_path
        )
        
        return y_train_pred, y_test_pred


def main():
    """主函数"""
    print("=" * 50)
    print("源CNN模型训练")
    print("=" * 50)
    
    # 获取配置
    data_config = DATA_CONFIG['source_cnn']
    
    # 加载数据
    print("加载数据...")
    X, y = load_and_prepare_data(
        file_path=data_config['data_file'],
        target_column=data_config['target_column'],
        drop_columns=[data_config['target_column']]  # 删除class列
    )
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"特征列数: {X.shape[1]}")
    print()
    
    # 准备数据
    print("准备CNN数据...")
    data_processor = DataProcessor()
    X_train, X_test, y_train, y_test = data_processor.prepare_cnn_data(
        X, y, 
        test_size=data_config['test_size'],
        random_state=data_config['random_state']
    )
    
    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
    print()
    
    # 创建和训练模型
    print("创建源CNN模型...")
    source_cnn = SourceCNN()
    
    # 训练模型
    model = source_cnn.train_model(X_train, y_train, X_test, y_test)
    
    # 评估模型
    print("评估模型性能...")
    metrics, y_test_pred = source_cnn.evaluate_model(X_test, y_test, 'cnn.h5')
    
    # 可视化结果
    print("生成可视化结果...")
    source_cnn.visualize_results(X_train, y_train, X_test, y_test, 'cnn.h5')
    
    # 保存预测结果
    print("保存预测结果...")
    results_df = pd.DataFrame({
        'predicted': y_test_pred,
        'actual': y_test
    })
    # results_df.to_csv('source_cnn_predictions.csv', index=False)
    print("预测结果已保存到 source_cnn_predictions.csv")
    
    print("=" * 50)
    print("源CNN模型训练完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
