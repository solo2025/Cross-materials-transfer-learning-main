"""
数据融合迁移学习脚本
使用多种机器学习算法进行数据融合和迁移学习
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import shap
from statistics import mean
import warnings
warnings.filterwarnings("ignore")

from config import DATA_CONFIG, MODEL_CONFIG
from utils import DataProcessor, ModelEvaluator, Visualizer, load_and_prepare_data


class MLModelTrainer:
    """机器学习模型训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.data_processor = DataProcessor()
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        self.models = {}
        self.results = {}
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, 
                          train1_size, cv_folds=5, test_runs=1):
        """
        训练随机森林模型
        
        Args:
            X_train (np.array): 训练特征
            y_train (np.array): 训练标签
            X_test (np.array): 测试特征
            y_test (np.array): 测试标签
            train1_size (int): 第一个训练集的大小
            cv_folds (int): 交叉验证折数
            test_runs (int): 测试运行次数
            
        Returns:
            dict: 训练结果
        """
        print("训练随机森林模型...")
        
        mae_list = []
        mse_list = []
        r2_list = []
        mape_list = []
        
        config = MODEL_CONFIG['ml_models']['random_forest']
        
        for i in range(test_runs):
            # 网格搜索最佳参数
            parameters = {
                'n_estimators': config['n_estimators'],
                'max_depth': config['max_depth'],
                'min_samples_split': config['min_samples_split'],
                'min_samples_leaf': config['min_samples_leaf']
            }
            
            rfr = RandomForestRegressor()
            gs = GridSearchCV(estimator=rfr, param_grid=parameters, cv=cv_folds)
            gs.fit(X_train, y_train)
            
            print(f"随机森林最佳超参数: {gs.best_params_}")
            
            # 使用最佳参数训练模型
            rfr_best = RandomForestRegressor(**gs.best_params_)
            rfr_best.fit(X_train, y_train)
            
            # 预测
            y_train_pred1 = rfr_best.predict(X_train[0:train1_size])
            y_train_pred2 = rfr_best.predict(X_train[train1_size:])
            y_test_pred = rfr_best.predict(X_test)
            
            # SHAP分析
            try:
                explainer = shap.Explainer(rfr_best.predict, X_train)
                shap_values = explainer(X_train)
                shap.summary_plot(shap_values, X_train, max_display=10, show=False)
            except Exception as e:
                print(f"SHAP分析失败: {e}")
            
            # 计算评估指标
            metrics = self.evaluator.calculate_metrics(y_test, y_test_pred)
            
            mae_list.append(metrics['mae'])
            mse_list.append(metrics['mse'])
            r2_list.append(metrics['r2'])
            mape_list.append(metrics['mape'])
            
            print(f"随机森林第{i+1}次循环, MSE:{metrics['mse']:.4f}, "
                  f"MAE:{metrics['mae']:.4f}, R²:{metrics['r2']:.4f}, "
                  f"MAPE:{metrics['mape']:.4f}")
        
        # 计算平均指标
        avg_metrics = {
            'mse': mean(mse_list),
            'mae': mean(mae_list),
            'r2': mean(r2_list),
            'mape': mean(mape_list)
        }
        
        print(f"随机森林平均指标 - MSE:{avg_metrics['mse']:.4f}, "
              f"MAE:{avg_metrics['mae']:.4f}, R²:{avg_metrics['r2']:.4f}, "
              f"MAPE:{avg_metrics['mape']:.4f}")
        
        return {
            'model': rfr_best,
            'metrics': avg_metrics,
            'predictions': {
                'train1': y_train_pred1,
                'train2': y_train_pred2,
                'test': y_test_pred
            }
        }
    
    def train_gaussian_process(self, X_train, y_train, X_test, y_test, 
                             train1_size, cv_folds=5, test_runs=1):
        """
        训练高斯过程回归模型
        
        Args:
            X_train (np.array): 训练特征
            y_train (np.array): 训练标签
            X_test (np.array): 测试特征
            y_test (np.array): 测试标签
            train1_size (int): 第一个训练集的大小
            cv_folds (int): 交叉验证折数
            test_runs (int): 测试运行次数
            
        Returns:
            dict: 训练结果
        """
        print("训练高斯过程回归模型...")
        
        mae_list = []
        mse_list = []
        r2_list = []
        mape_list = []
        
        config = MODEL_CONFIG['ml_models']['gaussian_process']
        
        for i in range(test_runs):
            # 准备核函数
            kernels = [None, DotProduct(), DotProduct() + WhiteKernel(), WhiteKernel()]
            
            # 网格搜索最佳参数
            parameters = {
                'normalize_y': config['normalize_y'],
                'kernel': kernels,
                'alpha': np.arange(0.001, 0.1, 0.001)
            }
            
            gpr = GaussianProcessRegressor()
            gs = GridSearchCV(estimator=gpr, param_grid=parameters, cv=cv_folds)
            gs.fit(X_train, y_train)
            
            print(f"高斯过程回归最佳超参数: {gs.best_params_}")
            print(f"高斯过程回归最佳分数: {gs.best_score_:.4f}")
            
            # 使用最佳参数训练模型
            gpr_best = GaussianProcessRegressor(**gs.best_params_)
            gpr_best.fit(X_train, y_train)
            
            # 预测
            y_train_pred1 = gpr_best.predict(X_train[0:train1_size])
            y_train_pred2 = gpr_best.predict(X_train[train1_size:])
            y_test_pred = gpr_best.predict(X_test)
            
            # SHAP分析
            try:
                explainer = shap.Explainer(gpr_best.predict, X_train)
                shap_values = explainer(X_train)
                shap.summary_plot(shap_values, X_train, max_display=10, show=False)
            except Exception as e:
                print(f"SHAP分析失败: {e}")
            
            # 计算评估指标
            metrics = self.evaluator.calculate_metrics(y_test, y_test_pred)
            
            mae_list.append(metrics['mae'])
            mse_list.append(metrics['mse'])
            r2_list.append(metrics['r2'])
            mape_list.append(metrics['mape'])
            
            print(f"高斯过程回归第{i+1}次循环, MSE:{metrics['mse']:.4f}, "
                  f"MAE:{metrics['mae']:.4f}, R²:{metrics['r2']:.4f}, "
                  f"MAPE:{metrics['mape']:.4f}")
        
        # 计算平均指标
        avg_metrics = {
            'mse': mean(mse_list),
            'mae': mean(mae_list),
            'r2': mean(r2_list),
            'mape': mean(mape_list)
        }
        
        print(f"高斯过程回归平均指标 - MSE:{avg_metrics['mse']:.4f}, "
              f"MAE:{avg_metrics['mae']:.4f}, R²:{avg_metrics['r2']:.4f}, "
              f"MAPE:{avg_metrics['mape']:.4f}")
        
        return {
            'model': gpr_best,
            'metrics': avg_metrics,
            'predictions': {
                'train1': y_train_pred1,
                'train2': y_train_pred2,
                'test': y_test_pred
            }
        }
    
    def train_svr(self, X_train, y_train, X_test, y_test, 
                 train1_size, cv_folds=10, test_runs=3):
        """
        训练支持向量回归模型
        
        Args:
            X_train (np.array): 训练特征
            y_train (np.array): 训练标签
            X_test (np.array): 测试特征
            y_test (np.array): 测试标签
            train1_size (int): 第一个训练集的大小
            cv_folds (int): 交叉验证折数
            test_runs (int): 测试运行次数
            
        Returns:
            dict: 训练结果
        """
        print("训练支持向量回归模型...")
        
        mae_list = []
        mse_list = []
        r2_list = []
        mape_list = []
        
        config = MODEL_CONFIG['ml_models']['svr']
        
        for i in range(test_runs):
            # 网格搜索最佳参数
            parameters = {
                'C': config['C'],
                'epsilon': np.arange(0.01, 1, 0.01)
            }
            
            svr = LinearSVR()
            gs = GridSearchCV(estimator=svr, param_grid=parameters, cv=cv_folds)
            gs.fit(X_train, y_train)
            
            print(f"SVR最佳超参数: {gs.best_params_}")
            
            # 使用最佳参数训练模型
            svr_best = LinearSVR(**gs.best_params_)
            svr_best.fit(X_train, y_train)
            
            # 预测
            y_train_pred1 = svr_best.predict(X_train[0:train1_size])
            y_train_pred2 = svr_best.predict(X_train[train1_size:])
            y_test_pred = svr_best.predict(X_test)
            
            # 计算评估指标
            metrics = self.evaluator.calculate_metrics(y_test, y_test_pred)
            
            mae_list.append(metrics['mae'])
            mse_list.append(metrics['mse'])
            r2_list.append(metrics['r2'])
            mape_list.append(metrics['mape'])
            
            print(f"SVR第{i+1}次循环, MSE:{metrics['mse']:.4f}, "
                  f"MAE:{metrics['mae']:.4f}, R²:{metrics['r2']:.4f}, "
                  f"MAPE:{metrics['mape']:.4f}")
        
        # 计算平均指标
        avg_metrics = {
            'mse': mean(mse_list),
            'mae': mean(mae_list),
            'r2': mean(r2_list),
            'mape': mean(mape_list)
        }
        
        print(f"SVR平均指标 - MSE:{avg_metrics['mse']:.4f}, "
              f"MAE:{avg_metrics['mae']:.4f}, R²:{avg_metrics['r2']:.4f}, "
              f"MAPE:{avg_metrics['mape']:.4f}")
        
        return {
            'model': svr_best,
            'metrics': avg_metrics,
            'predictions': {
                'train1': y_train_pred1,
                'train2': y_train_pred2,
                'test': y_test_pred
            }
        }
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, 
                     train1_size, cv_folds=5, test_runs=1):
        """
        训练XGBoost模型
        
        Args:
            X_train (np.array): 训练特征
            y_train (np.array): 训练标签
            X_test (np.array): 测试特征
            y_test (np.array): 测试标签
            train1_size (int): 第一个训练集的大小
            cv_folds (int): 交叉验证折数
            test_runs (int): 测试运行次数
            
        Returns:
            dict: 训练结果
        """
        print("训练XGBoost模型...")
        
        mae_list = []
        mse_list = []
        r2_list = []
        mape_list = []
        
        config = MODEL_CONFIG['ml_models']['xgboost']
        
        for i in range(test_runs):
            # 网格搜索最佳参数
            parameters = {
                'n_estimators': config['n_estimators'],
                'max_depth': config['max_depth'],
                'min_child_weight': config['min_child_weight'],
                'gamma': config['gamma'],
                'subsample': config['subsample'],
                'colsample_bytree': config['colsample_bytree'],
                'reg_lambda': config['reg_lambda'],
                'reg_alpha': config['reg_alpha']
            }
            
            xgbreg = xgb.XGBRegressor()
            gs = GridSearchCV(estimator=xgbreg, param_grid=parameters, cv=cv_folds)
            gs.fit(X_train, y_train)
            
            print(f"XGBoost最佳超参数: {gs.best_params_}")
            
            # 使用最佳参数训练模型
            xgb_best = xgb.XGBRegressor(**gs.best_params_)
            xgb_best.fit(X_train, y_train)
            
            # 预测
            y_train_pred1 = xgb_best.predict(X_train[0:train1_size])
            y_train_pred2 = xgb_best.predict(X_train[train1_size:])
            y_test_pred = xgb_best.predict(X_test)
            
            # 计算评估指标
            metrics = self.evaluator.calculate_metrics(y_test, y_test_pred)
            
            mae_list.append(metrics['mae'])
            mse_list.append(metrics['mse'])
            r2_list.append(metrics['r2'])
            mape_list.append(metrics['mape'])
            
            print(f"XGBoost第{i+1}次循环, MSE:{metrics['mse']:.4f}, "
                  f"MAE:{metrics['mae']:.4f}, R²:{metrics['r2']:.4f}, "
                  f"MAPE:{metrics['mape']:.4f}")
        
        # 计算平均指标
        avg_metrics = {
            'mse': mean(mse_list),
            'mae': mean(mae_list),
            'r2': mean(r2_list),
            'mape': mean(mape_list)
        }
        
        print(f"XGBoost平均指标 - MSE:{avg_metrics['mse']:.4f}, "
              f"MAE:{avg_metrics['mae']:.4f}, R²:{avg_metrics['r2']:.4f}, "
              f"MAPE:{avg_metrics['mape']:.4f}")
        
        return {
            'model': xgb_best,
            'metrics': avg_metrics,
            'predictions': {
                'train1': y_train_pred1,
                'train2': y_train_pred2,
                'test': y_test_pred
            }
        }
    
    def train_adaboost(self, X_train, y_train, X_test, y_test, 
                      train1_size, cv_folds=5, test_runs=1):
        """
        训练AdaBoost模型
        
        Args:
            X_train (np.array): 训练特征
            y_train (np.array): 训练标签
            X_test (np.array): 测试特征
            y_test (np.array): 测试标签
            train1_size (int): 第一个训练集的大小
            cv_folds (int): 交叉验证折数
            test_runs (int): 测试运行次数
            
        Returns:
            dict: 训练结果
        """
        print("训练AdaBoost模型...")
        
        mae_list = []
        mse_list = []
        r2_list = []
        mape_list = []
        
        config = MODEL_CONFIG['ml_models']['adaboost']
        
        for i in range(test_runs):
            # 网格搜索最佳参数
            parameters = {
                'n_estimators': config['n_estimators']
            }
            
            ada = AdaBoostRegressor()
            gs = GridSearchCV(estimator=ada, param_grid=parameters, cv=cv_folds)
            gs.fit(X_train, y_train)
            
            print(f"AdaBoost最佳超参数: {gs.best_params_}")
            
            # 使用最佳参数训练模型
            ada_best = AdaBoostRegressor(**gs.best_params_)
            ada_best.fit(X_train, y_train)
            
            # 预测
            y_train_pred1 = ada_best.predict(X_train[0:train1_size])
            y_train_pred2 = ada_best.predict(X_train[train1_size:])
            y_test_pred = ada_best.predict(X_test)
            
            # 计算评估指标
            metrics = self.evaluator.calculate_metrics(y_test, y_test_pred)
            
            mae_list.append(metrics['mae'])
            mse_list.append(metrics['mse'])
            r2_list.append(metrics['r2'])
            mape_list.append(metrics['mape'])
            
            print(f"AdaBoost第{i+1}次循环, MSE:{metrics['mse']:.4f}, "
                  f"MAE:{metrics['mae']:.4f}, R²:{metrics['r2']:.4f}, "
                  f"MAPE:{metrics['mape']:.4f}")
        
        # 计算平均指标
        avg_metrics = {
            'mse': mean(mse_list),
            'mae': mean(mae_list),
            'r2': mean(r2_list),
            'mape': mean(mape_list)
        }
        
        print(f"AdaBoost平均指标 - MSE:{avg_metrics['mse']:.4f}, "
              f"MAE:{avg_metrics['mae']:.4f}, R²:{avg_metrics['r2']:.4f}, "
              f"MAPE:{avg_metrics['mape']:.4f}")
        
        return {
            'model': ada_best,
            'metrics': avg_metrics,
            'predictions': {
                'train1': y_train_pred1,
                'train2': y_train_pred2,
                'test': y_test_pred
            }
        }
    
    def visualize_model_results(self, model_name, predictions, y_train1, y_train2, y_test):
        """
        可视化模型结果
        
        Args:
            model_name (str): 模型名称
            predictions (dict): 预测结果字典
            y_train1 (np.array): 训练集1真实值
            y_train2 (np.array): 训练集2真实值
            y_test (np.array): 测试集真实值
        """
        self.visualizer.plot_data_fusion_predictions(
            predictions['train1'], y_train1,
            predictions['train2'], y_train2,
            predictions['test'], y_test,
            model_name=model_name,
            save_path=f'{model_name}_predictions.png'
        )


def prepare_fusion_data(train_file1, train_file2, test_file, target_column):
    """
    准备数据融合数据
    
    Args:
        train_file1 (str): 训练数据文件1路径
        train_file2 (str): 训练数据文件2路径
        test_file (str): 测试数据文件路径
        target_column (str): 目标列名
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, train1_size)
    """
    # 读取训练数据
    df_train1 = pd.read_csv(train_file1)
    df_train2 = pd.read_csv(train_file2)
    df_train = pd.concat([df_train1, df_train2])
    df_train = df_train.iloc[:, 1:]  # 删除第一列
    
    # 读取测试数据
    df_test = pd.read_csv(test_file)
    df_test = df_test.iloc[:, 1:]  # 删除第一列
    
    # 合并数据用于统一标准化
    df_all = pd.concat([df_train, df_test], axis=0)
    train_size = df_train.shape[0]
    train1_size = df_train1.shape[0]
    
    # 分离特征和目标
    X_all = df_all.drop(target_column, axis=1)
    y_all = df_all[target_column]
    
    # 标准化特征
    data_processor = DataProcessor()
    X_all_scaled = data_processor.scaler.fit_transform(X_all)
    
    # 分离训练和测试数据
    X_train = X_all_scaled[0:train_size]
    X_test = X_all_scaled[train_size:]
    y_train = y_all[0:train_size]
    y_test = y_all[train_size:]
    
    return X_train, X_test, y_train, y_test, train1_size


def main():
    """主函数"""
    print("=" * 50)
    print("数据融合迁移学习")
    print("=" * 50)
    
    # 获取配置
    data_config = DATA_CONFIG['data_fusion']
    
    # 准备数据
    print("准备数据融合数据...")
    X_train, X_test, y_train, y_test, train1_size = prepare_fusion_data(
        train_file1=data_config['train_file1'],
        train_file2=data_config['train_file2'],
        test_file=data_config['test_file'],
        target_column=data_config['target_column']
    )
    
    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"训练集1大小: {train1_size}")
    print()
    
    # 创建训练器
    trainer = MLModelTrainer()
    
    # 分离训练集1和训练集2的真实值
    y_train1 = y_train[0:train1_size]
    y_train2 = y_train[train1_size:]
    
    # 训练各种模型
    models_to_train = [
        ('RandomForest', trainer.train_random_forest),
        ('GaussianProcess', trainer.train_gaussian_process),
        ('SVR', trainer.train_svr),
        ('XGBoost', trainer.train_xgboost),
        ('AdaBoost', trainer.train_adaboost)
    ]
    
    results = {}
    
    for model_name, train_func in models_to_train:
        print(f"\n{'='*20} {model_name} {'='*20}")
        
        # 训练模型
        result = train_func(X_train, y_train, X_test, y_test, train1_size)
        results[model_name] = result
        
        # 可视化结果
        trainer.visualize_model_results(
            model_name, 
            result['predictions'], 
            y_train1, y_train2, y_test
        )
        
        print(f"{model_name} 训练完成!")
    
    # 汇总结果
    print("\n" + "="*50)
    print("所有模型结果汇总")
    print("="*50)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:15} - MSE:{metrics['mse']:.4f}, "
              f"MAE:{metrics['mae']:.4f}, R²:{metrics['r2']:.4f}, "
              f"MAPE:{metrics['mape']:.4f}")
    
    print("\n" + "="*50)
    print("数据融合迁移学习完成!")
    print("="*50)


if __name__ == "__main__":
    main()
