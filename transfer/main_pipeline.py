"""
主流程脚本 - 合金材料机器学习流水线
整合所有功能模块，提供统一的接口
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import torch

# 添加core模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.config_manager import ConfigManager
from core.models import VAMNet, AMFeatureNet, TransferLearningNet, ModelManager
from core.data_processing import DataProcessor, FeatureSelector, DataAnalyzer
from core.training import ModelTrainer, ModelEvaluator
from core.visualization import DataVisualizer, ModelExplainer, ReportGenerator
from core.alloy_design import AlloyDesigner, GeneticOptimizer


class AlloyMLPipeline:
    """合金机器学习流水线"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化流水线"""
        
        # 加载配置
        self.config_manager = ConfigManager(config_path or "config.json")
        self.device = torch.device(self.config_manager.get_device())
        
        # 初始化组件
        self.data_processor = DataProcessor(random_state=self.config_manager.get_random_state())
        self.feature_selector = FeatureSelector(random_state=self.config_manager.get_random_state())
        self.data_analyzer = DataAnalyzer()
        self.model_trainer = ModelTrainer(device=self.device, random_state=self.config_manager.get_random_state())
        self.model_evaluator = ModelEvaluator()
        self.visualizer = DataVisualizer()
        self.explainer = ModelExplainer(device=self.device)
        self.report_generator = ReportGenerator()
        self.model_manager = ModelManager(device=self.device)
        
        # 创建输出目录
        self._create_output_directories()
    
    
    def _create_output_directories(self):
        """创建输出目录"""
        
        for dir_name in self.config_manager.get('output_dirs', {}).values():
            os.makedirs(dir_name, exist_ok=True)
    
    def analyze_data(self, data_type: str = 'vam') -> Dict[str, Any]:
        """数据分析"""
        
        print(f"开始分析 {data_type.upper()} 数据...")
        
        # 加载数据
        data_path = self.config_manager.get_data_path(f'{data_type}_data')
        data = self.data_processor.load_data(data_path)
        
        # 成分分析
        composition_cols = self.config_manager.get_feature_columns('composition')
        composition_analysis = self.data_analyzer.analyze_composition_distribution(data, composition_cols)
        
        # 相关性分析
        all_features = (self.config_manager.get_feature_columns('parameters') + 
                       self.config_manager.get_feature_columns(f'{data_type}_process') + 
                       self.config_manager.get_feature_columns('properties'))
        
        corr_matrix = self.data_analyzer.calculate_correlation_matrix(data, all_features)
        
        # 可视化
        plots_dir = self.config_manager.get_output_dir('plots')
        self.visualizer.plot_composition_distribution(
            data, composition_cols, 
            title=f'{data_type.upper()} 成分分布',
            save_path=f"{plots_dir}/{data_type}_composition_distribution.png"
        )
        
        self.visualizer.plot_correlation_matrix(
            data, all_features,
            title=f'{data_type.upper()} 相关性矩阵',
            save_path=f"{plots_dir}/{data_type}_correlation_matrix.png"
        )
        
        # 生成报告
        report = self.report_generator.generate_data_report(
            data, composition_cols, self.config_manager.get_feature_columns('properties')
        )
        
        # 保存结果
        results = {
            'data_summary': data.describe(),
            'composition_analysis': composition_analysis,
            'correlation_matrix': corr_matrix,
            'report': report
        }
        
        # 保存到文件
        results_dir = self.config_manager.get_output_dir('results')
        pd.DataFrame(composition_analysis['element_counts'], index=[0]).to_csv(
            f"{results_dir}/{data_type}_element_counts.csv"
        )
        
        corr_matrix.to_csv(f"{results_dir}/{data_type}_correlation_matrix.csv")
        
        reports_dir = self.config_manager.get_output_dir('reports')
        with open(f"{reports_dir}/{data_type}_data_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"{data_type.upper()} 数据分析完成")
        return results
    
    def train_vam_model(self) -> Dict[str, Any]:
        """训练VAM模型"""
        
        print("开始训练VAM模型...")
        
        # 加载数据
        data_path = self.config['data_paths']['vam_data']
        data = self.data_processor.load_data(data_path)
        
        # 准备特征
        X, y = self.data_processor.prepare_vam_features(
            data,
            self.config['feature_columns']['composition'],
            self.config['feature_columns']['vam_process'],
            self.config['feature_columns']['properties'],
            self.config['feature_columns']['parameters']
        )
        
        # 分割数据
        X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)
        
        # 特征标准化
        X_train_scaled, X_test_scaled = self.data_processor.scale_features(
            X_train, X_test, scaler_name='vam_scaler'
        )
        
        # 保存标准化器
        self.data_processor.save_scaler(
            'vam_scaler', 
            f"{self.config['output_dirs']['models']}/vam_scaler.pkl"
        )
        
        # 创建模型
        model_config = self.config['model_config']['vam_net']
        model = VAMNet(**model_config)
        
        # 训练模型
        training_config = self.config['model_config']['training']
        training_results = self.model_trainer.train_model(
            model, X_train_scaled, y_train, X_test_scaled, y_test,
            model_name='vam_model', **training_config
        )
        
        # 评估模型
        metrics = self.model_trainer.evaluate_model(model, X_test_scaled, y_test, 'vam_model')
        
        # 可视化
        y_pred = self.model_trainer.predict('vam_model', torch.tensor(X_test_scaled, dtype=torch.float))
        y_pred = y_pred.cpu().numpy()
        
        self.model_trainer.plot_training_history(
            'vam_model',
            save_path=f"{self.config['output_dirs']['plots']}/vam_training_history.png"
        )
        
        self.model_trainer.plot_predictions(
            y_test, y_pred, 'VAM模型',
            save_path=f"{self.config['output_dirs']['plots']}/vam_predictions.png"
        )
        
        # 保存模型
        self.model_trainer.save_model(
            model, 'vam_model', 
            self.config['output_dirs']['models']
        )
        
        # 生成报告
        training_history = self.model_trainer.training_history['vam_model']
        report = self.report_generator.generate_model_report(
            metrics, training_history, 'VAM模型'
        )
        
        with open(f"{self.config['output_dirs']['reports']}/vam_model_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("VAM模型训练完成")
        return {
            'training_results': training_results,
            'metrics': metrics,
            'model': model
        }
    
    def train_am_model(self) -> Dict[str, Any]:
        """训练AM模型（迁移学习）"""
        
        print("开始训练AM模型（迁移学习）...")
        
        # 加载VAM模型
        vam_model_path = f"{self.config['output_dirs']['models']}/vam_model_best.pth"
        if not os.path.exists(vam_model_path):
            raise FileNotFoundError("VAM模型不存在，请先训练VAM模型")
        
        # 加载VAM模型
        vam_model = VAMNet(**self.config['model_config']['vam_net'])
        vam_model.load_state_dict(torch.load(vam_model_path, map_location=self.device))
        
        # 加载AM数据
        train_data = self.data_processor.load_data(self.config['data_paths']['am_data'])
        valid_data = self.data_processor.load_data(self.config['data_paths']['am_valid'])
        
        # 准备AM特征
        X_train, y_train = self.data_processor.prepare_am_features(
            train_data,
            self.config['feature_columns']['composition'],
            self.config['feature_columns']['am_process'],
            self.config['feature_columns']['properties'],
            self.config['feature_columns']['parameters']
        )
        
        X_valid, y_valid = self.data_processor.prepare_am_features(
            valid_data,
            self.config['feature_columns']['composition'],
            self.config['feature_columns']['am_process'],
            self.config['feature_columns']['properties'],
            self.config['feature_columns']['parameters']
        )
        
        # 特征标准化
        X_train_scaled, X_valid_scaled = self.data_processor.scale_features(
            X_train, X_valid, scaler_name='am_scaler'
        )
        
        # 保存标准化器
        self.data_processor.save_scaler(
            'am_scaler',
            f"{self.config['output_dirs']['models']}/am_scaler.pkl"
        )
        
        # 创建迁移学习模型
        am_feature_net = AMFeatureNet(**self.config['model_config']['am_feature_net'])
        transfer_model = TransferLearningNet(am_feature_net, vam_model)
        
        # 训练模型
        training_config = self.config['model_config']['training']
        training_results = self.model_trainer.train_model(
            transfer_model, X_train_scaled, y_train, X_valid_scaled, y_valid,
            model_name='am_model', **training_config
        )
        
        # 评估模型
        metrics = self.model_trainer.evaluate_model(transfer_model, X_valid_scaled, y_valid, 'am_model')
        
        # 可视化
        y_pred = self.model_trainer.predict('am_model', torch.tensor(X_valid_scaled, dtype=torch.float))
        y_pred = y_pred.cpu().numpy()
        
        self.model_trainer.plot_training_history(
            'am_model',
            save_path=f"{self.config['output_dirs']['plots']}/am_training_history.png"
        )
        
        self.model_trainer.plot_predictions(
            y_valid, y_pred, 'AM模型',
            save_path=f"{self.config['output_dirs']['plots']}/am_predictions.png"
        )
        
        # 保存模型
        self.model_trainer.save_model(
            transfer_model, 'am_model',
            self.config['output_dirs']['models']
        )
        
        # 生成报告
        training_history = self.model_trainer.training_history['am_model']
        report = self.report_generator.generate_model_report(
            metrics, training_history, 'AM模型'
        )
        
        with open(f"{self.config['output_dirs']['reports']}/am_model_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("AM模型训练完成")
        return {
            'training_results': training_results,
            'metrics': metrics,
            'model': transfer_model
        }
    
    def design_alloy(self, num_samples: int = 1000, 
                    optimization: bool = False) -> Dict[str, Any]:
        """合金设计"""
        
        print("开始合金设计...")
        
        # 定义成分约束
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
        
        # 定义工艺参数
        process_params = {
            'LED': 1.12,
            'AED': 18.67,
            'VED': 466.7,
            'C(0)/T(1)': 0
        }
        
        # 创建预测函数
        def predict_property(alloy_data):
            # 这里需要加载训练好的模型进行预测
            # 简化版本，实际应用中需要加载真实模型
            return np.random.normal(1000, 200, (len(alloy_data), 1))
        
        # 创建合金设计器
        designer = AlloyDesigner(
            self.config['feature_columns']['composition'],
            composition_constraints,
            predict_property
        )
        
        if optimization:
            # 使用遗传算法优化
            results = designer.optimize_alloy(
                population_size=100,
                generations=50,
                process_params=process_params
            )
            
            # 保存优化结果
            os.makedirs(f"{self.config['output_dirs']['results']}/optimization", exist_ok=True)
            
            # 保存Pareto前沿
            pareto_front = results['evolution_history']['pareto_front'][-1]
            pareto_data = []
            
            for ind in pareto_front:
                composition = [x * 100 for x in ind]
                pareto_data.append(composition + list(ind.fitness.values))
            
            pareto_df = pd.DataFrame(pareto_data,
                                   columns=self.config['feature_columns']['composition'] + ['strength', 'density'])
            pareto_df.to_csv(f"{self.config['output_dirs']['results']}/optimization/pareto_front.csv", index=False)
            
            print("合金优化完成")
            return results
        else:
            # 随机生成合金
            results = designer.design_alloy(num_samples, process_params)
            
            # 保存结果
            results.to_csv(f"{self.config['output_dirs']['results']}/alloy_design.csv", index=False)
            
            print("合金设计完成")
            return {'design_results': results}
    
    def run_full_pipeline(self):
        """运行完整流水线"""
        
        print("开始运行完整流水线...")
        
        # 1. 数据分析
        vam_analysis = self.analyze_data('vam')
        am_analysis = self.analyze_data('am')
        
        # 2. 模型训练
        vam_results = self.train_vam_model()
        am_results = self.train_am_model()
        
        # 3. 合金设计
        design_results = self.design_alloy(optimization=True)
        
        print("完整流水线运行完成")
        
        return {
            'vam_analysis': vam_analysis,
            'am_analysis': am_analysis,
            'vam_results': vam_results,
            'am_results': am_results,
            'design_results': design_results
        }


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='合金材料机器学习流水线')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['analyze', 'train', 'design', 'full'], 
                       default='full', help='运行模式')
    parser.add_argument('--data_type', type=str, choices=['vam', 'am'], 
                       default='vam', help='数据类型')
    
    args = parser.parse_args()
    
    # 创建流水线
    pipeline = AlloyMLPipeline(args.config)
    
    # 根据模式运行
    if args.mode == 'analyze':
        pipeline.analyze_data(args.data_type)
    elif args.mode == 'train':
        if args.data_type == 'vam':
            pipeline.train_vam_model()
        else:
            pipeline.train_am_model()
    elif args.mode == 'design':
        pipeline.design_alloy(optimization=True)
    elif args.mode == 'full':
        pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()
