"""
简化使用示例
演示如何使用优化后的模块化代码
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加core模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.data_processing import DataProcessor, DataAnalyzer
from core.visualization import DataVisualizer
from core.models import VAMNet
from core.training import ModelTrainer


def example_data_analysis():
    """数据分析示例"""
    
    print("=== 数据分析示例 ===")
    
    # 初始化组件
    processor = DataProcessor()
    analyzer = DataAnalyzer()
    visualizer = DataVisualizer()
    
    # 加载数据
    data_path = 'data/VAM_LWRHEAS_property_AlNbTiVZrCrMoHfTaW_20241202-clean-RT.csv'
    if os.path.exists(data_path):
        data = processor.load_data(data_path)
        print(f"数据加载成功，形状: {data.shape}")
        
        # 成分分析
        composition_cols = ['x(Al)', 'x(Nb)', 'x(Ti)', 'x(V)', 'x(Zr)', 'x(Cr)', 'x(Mo)', 'x(Hf)', 'x(Ta)']
        composition_analysis = analyzer.analyze_composition_distribution(data, composition_cols)
        
        print("成分分析结果:")
        for element, count in composition_analysis['element_counts'].items():
            print(f"  {element}: {count} 个非零样本")
        
        # 可视化
        visualizer.plot_composition_distribution(data, composition_cols, "成分分布示例")
        
    else:
        print(f"数据文件不存在: {data_path}")


def example_model_training():
    """模型训练示例"""
    
    print("\n=== 模型训练示例 ===")
    
    # 初始化组件
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # 创建示例数据
    np.random.seed(42)
    X = np.random.randn(100, 11)  # 11个特征
    y = np.random.randn(100, 1)   # 1个目标
    
    # 分割数据
    X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2)
    
    # 标准化
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    
    # 创建模型
    model = VAMNet(input_size=11, hidden_sizes=[64, 32])
    
    # 训练模型
    training_results = trainer.train_model(
        model, X_train_scaled, y_train, X_test_scaled, y_test,
        epochs=100, learning_rate=0.01, model_name='example_model'
    )
    
    print(f"训练完成，最佳轮次: {training_results['best_epoch']}")
    
    # 评估模型
    metrics = trainer.evaluate_model(model, X_test_scaled, y_test, 'example_model')
    print(f"R² 分数: {metrics['r2']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")


def example_alloy_generation():
    """合金生成示例"""
    
    print("\n=== 合金生成示例 ===")
    
    from core.alloy_design import AlloyGenerator
    
    # 定义元素和约束
    element_list = ['x(Al)', 'x(Nb)', 'x(Ti)', 'x(V)', 'x(Zr)', 'x(Cr)', 'x(Mo)', 'x(Hf)', 'x(Ta)']
    composition_constraints = {
        'x(Al)': (0, 20),
        'x(Nb)': (0, 35),
        'x(Ti)': (0, 40),
        'x(V)': (0, 40),
        'x(Zr)': (0, 35),
        'x(Cr)': (0, 20),
        'x(Mo)': (0, 20),
        'x(Hf)': (0, 15),
        'x(Ta)': (0, 15)
    }
    
    process_params = {
        'LED': 1.12,
        'AED': 18.67,
        'VED': 466.7,
        'C(0)/T(1)': 0
    }
    
    # 创建生成器
    generator = AlloyGenerator(element_list)
    
    # 生成合金
    data, element_data = generator.generate_random_alloy(10, composition_constraints, process_params)
    
    print(f"生成了 {len(data)} 个合金样本")
    print("前3个样本:")
    for i in range(min(3, len(data))):
        print(f"  样本 {i+1}: {data[i][:9]}")  # 只显示成分部分


def main():
    """主函数"""
    
    print("合金材料机器学习流水线 - 使用示例")
    print("=" * 50)
    
    # 运行示例
    example_data_analysis()
    example_model_training()
    example_alloy_generation()
    
    print("\n示例运行完成！")
    print("\n要运行完整流水线，请使用:")
    print("python main_pipeline.py --mode full")


if __name__ == '__main__':
    main()
