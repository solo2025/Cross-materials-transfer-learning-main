"""
模型训练和评估模块
包含训练、验证、评估等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, 
                           explained_variance_score, max_error, mean_absolute_percentage_error)
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from datetime import datetime


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, device: str = 'cpu', random_state: int = 42):
        self.device = device
        self.random_state = random_state
        self.training_history = {}
        self.best_models = {}
        
        # 设置随机种子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def train_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 1000, learning_rate: float = 0.001,
                   weight_decay: float = 0.001, patience: int = 50,
                   model_name: str = 'model') -> Dict[str, Any]:
        """训练模型"""
        
        # 转换为张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float).to(self.device)
        
        # 移动到设备
        model.to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 训练历史
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        print(f"开始训练模型: {model_name}")
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            optimizer.zero_grad()
            
            train_pred = model(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)
            
            # 记录损失
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            # 早停检查
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型
                self.best_models[model_name] = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, "
                      f"Val Loss = {val_loss.item():.4f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"早停于第 {epoch} 轮，最佳验证损失: {best_val_loss:.4f}")
                break
        
        # 保存训练历史
        self.training_history[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }
        
        return {
            'model_name': model_name,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_epochs': len(train_losses)
        }
    
    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = 'model') -> Dict[str, float]:
        """评估模型"""
        
        model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy()
        
        # 计算各种评估指标
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'evs': explained_variance_score(y_test, y_pred),
            'max_error': max_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred)
        }
        
        return metrics
    
    def plot_training_history(self, model_name: str, save_path: Optional[str] = None):
        """绘制训练历史"""
        
        if model_name not in self.training_history:
            raise ValueError(f"模型 {model_name} 的训练历史不存在")
        
        history = self.training_history[model_name]
        
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_losses'], label='训练损失', alpha=0.8)
        plt.plot(history['val_losses'], label='验证损失', alpha=0.8)
        plt.axvline(x=history['best_epoch'], color='red', linestyle='--', 
                   label=f'最佳轮次 ({history["best_epoch"]})')
        
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title(f'{model_name} 训练历史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str = 'model', save_path: Optional[str] = None):
        """绘制预测结果"""
        
        plt.figure(figsize=(8, 8))
        
        # 计算R²和MAE
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # 绘制散点图
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # 绘制理想线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{model_name} 预测结果\nR² = {r2:.4f}, MAE = {mae:.4f}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, model: nn.Module, model_name: str, save_dir: str):
        """保存模型"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        model_path = os.path.join(save_dir, f'{model_name}_best.pth')
        torch.save(self.best_models[model_name], model_path)
        
        # 保存训练历史
        history_path = os.path.join(save_dir, f'{model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history[model_name], f, indent=2)
        
        print(f"模型 {model_name} 已保存到 {save_dir}")
    
    def load_model(self, model: nn.Module, model_name: str, model_path: str):
        """加载模型"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        print(f"模型 {model_name} 已从 {model_path} 加载")


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def cross_validate(self, model_class: nn.Module, X: np.ndarray, y: np.ndarray,
                      cv_folds: int = 5, **train_kwargs) -> Dict[str, List[float]]:
        """交叉验证"""
        
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        fold_scores = {
            'r2': [],
            'mae': [],
            'mse': [],
            'rmse': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"交叉验证 - 第 {fold + 1}/{cv_folds} 折")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建新模型实例
            model = model_class()
            
            # 训练模型
            trainer = ModelTrainer()
            trainer.train_model(model, X_train, y_train, X_val, y_val, **train_kwargs)
            
            # 评估模型
            metrics = trainer.evaluate_model(model, X_val, y_val)
            
            for metric in fold_scores.keys():
                fold_scores[metric].append(metrics[metric])
        
        # 计算平均分数
        avg_scores = {metric: np.mean(scores) for metric, scores in fold_scores.items()}
        std_scores = {metric: np.std(scores) for metric, scores in fold_scores.items()}
        
        self.evaluation_results['cross_validation'] = {
            'fold_scores': fold_scores,
            'average_scores': avg_scores,
            'std_scores': std_scores
        }
        
        return self.evaluation_results['cross_validation']
    
    def generate_evaluation_report(self, metrics: Dict[str, float], 
                                 model_name: str = 'model') -> str:
        """生成评估报告"""
        
        report = f"""
模型评估报告 - {model_name}
{'=' * 50}

性能指标:
- R² 决定系数: {metrics['r2']:.4f}
- 平均绝对误差 (MAE): {metrics['mae']:.4f}
- 均方误差 (MSE): {metrics['mse']:.4f}
- 均方根误差 (RMSE): {metrics['rmse']:.4f}
- 解释方差分数 (EVS): {metrics['evs']:.4f}
- 最大误差: {metrics['max_error']:.4f}
- 平均绝对百分比误差 (MAPE): {metrics['mape']:.4f}

评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
