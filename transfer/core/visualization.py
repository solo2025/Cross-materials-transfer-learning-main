"""
可视化和分析模块
包含数据可视化、模型解释等功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import shap
import torch
import os
from typing import List, Dict, Optional, Tuple, Any


class DataVisualizer:
    """数据可视化器"""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_correlation_matrix(self, data: pd.DataFrame, columns: List[str],
                              title: str = '相关性矩阵', save_path: Optional[str] = None):
        """绘制相关性矩阵"""
        
        corr_matrix = data[columns].corr(method='spearman')
        
        plt.figure(figsize=self.figsize)
        
        # 创建热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='viridis', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               title: str = '特征重要性', save_path: Optional[str] = None):
        """绘制特征重要性"""
        
        plt.figure(figsize=(10, 6))
        
        # 按重要性排序
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        # 绘制水平条形图
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], 
                       color=self.colors[0], alpha=0.8)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.xlabel('重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_composition_distribution(self, data: pd.DataFrame, 
                                    composition_cols: List[str],
                                    title: str = '成分分布', save_path: Optional[str] = None):
        """绘制成分分布"""
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(composition_cols):
            if i < len(axes):
                # 过滤掉0值
                non_zero_data = data[data[col] > 0][col]
                
                axes[i].hist(non_zero_data, bins=20, alpha=0.7, color=self.colors[i % len(self.colors)])
                axes[i].set_title(f'{col} 分布', fontsize=10)
                axes[i].set_xlabel('含量 (%)', fontsize=8)
                axes[i].set_ylabel('频次', fontsize=8)
                axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(composition_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_tsne_visualization(self, X: np.ndarray, labels: List[str],
                               title: str = 't-SNE 可视化', save_path: Optional[str] = None):
        """绘制t-SNE可视化"""
        
        # 执行t-SNE
        tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X)
        
        # 创建DataFrame
        df_tsne = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])
        df_tsne['Label'] = labels
        
        plt.figure(figsize=self.figsize)
        
        # 绘制不同标签的点
        unique_labels = list(set(labels))
        for i, label in enumerate(unique_labels):
            mask = df_tsne['Label'] == label
            plt.scatter(df_tsne[mask]['t-SNE 1'], df_tsne[mask]['t-SNE 2'], 
                       c=self.colors[i % len(self.colors)], label=label, alpha=0.7, s=50)
        
        plt.xlabel('t-SNE 维度 1', fontsize=12)
        plt.ylabel('t-SNE 维度 2', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return df_tsne
    
    def plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = '预测结果', save_path: Optional[str] = None):
        """绘制预测散点图"""
        
        from sklearn.metrics import r2_score, mean_absolute_error
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        plt.figure(figsize=(8, 8))
        
        # 绘制散点图
        plt.scatter(y_true, y_pred, alpha=0.6, s=50, color=self.colors[0])
        
        # 绘制理想线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线')
        
        plt.xlabel('真实值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.title(f'{title}\nR² = {r2:.4f}, MAE = {mae:.4f}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ModelExplainer:
    """模型解释器"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.explanations = {}
    
    def shap_explain(self, model: torch.nn.Module, X: torch.Tensor, 
                    feature_names: List[str], model_name: str = 'model',
                    save_dir: Optional[str] = None) -> Dict[str, Any]:
        """SHAP解释"""
        
        # 创建SHAP解释器
        explainer = shap.DeepExplainer(model, X)
        shap_values = explainer.shap_values(X, check_additivity=False)
        
        # 调整形状
        if len(shap_values.shape) == 3:
            shap_values = shap_values.squeeze()
        
        # 转换为numpy
        X_numpy = X.numpy()
        
        # 创建SHAP图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, features=X_numpy, feature_names=feature_names, show=False)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'shap_summary_{model_name}.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # 创建条形图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, features=X_numpy, feature_names=feature_names, 
                         plot_type="bar", show=False)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'shap_bar_{model_name}.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # 保存SHAP值
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        
        if save_dir:
            shap_df.to_csv(os.path.join(save_dir, f'shap_values_{model_name}.csv'), index=False)
        
        # 保存解释结果
        explanation = {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'shap_df': shap_df
        }
        
        self.explanations[model_name] = explanation
        
        return explanation
    
    def generate_feature_importance_analysis(self, shap_values: np.ndarray,
                                           feature_names: List[str]) -> pd.DataFrame:
        """生成特征重要性分析"""
        
        # 计算平均绝对SHAP值
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_ABS_SHAP': mean_abs_shap
        }).sort_values('Mean_ABS_SHAP', ascending=False)
        
        return importance_df
    
    def plot_feature_interaction(self, shap_values: np.ndarray, feature_names: List[str],
                               top_n: int = 10, save_path: Optional[str] = None):
        """绘制特征交互图"""
        
        # 计算特征间的相关性
        shap_corr = np.corrcoef(shap_values.T)
        
        # 选择最重要的特征
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:]
        
        # 创建子矩阵
        top_corr = shap_corr[np.ix_(top_indices, top_indices)]
        top_names = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(top_corr, annot=True, cmap='coolwarm', center=0,
                   xticklabels=top_names, yticklabels=top_names)
        
        plt.title(f'Top {top_n} 特征交互矩阵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.reports = {}
    
    def generate_data_report(self, data: pd.DataFrame, 
                           composition_cols: List[str],
                           property_cols: List[str]) -> str:
        """生成数据报告"""
        
        report = f"""
数据报告
{'=' * 50}

数据集概览:
- 总样本数: {len(data)}
- 特征数量: {len(data.columns)}
- 成分特征: {len(composition_cols)}
- 属性特征: {len(property_cols)}

成分统计:
"""
        
        for col in composition_cols:
            non_zero_count = (data[col] > 0).sum()
            report += f"- {col}: {non_zero_count} 个非零样本\n"
        
        report += f"""
属性统计:
"""
        
        for col in property_cols:
            stats = data[col].describe()
            report += f"- {col}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}\n"
        
        return report
    
    def generate_model_report(self, metrics: Dict[str, float],
                           training_history: Dict[str, Any],
                           model_name: str = 'model') -> str:
        """生成模型报告"""
        
        report = f"""
模型报告 - {model_name}
{'=' * 50}

性能指标:
- R² 决定系数: {metrics['r2']:.4f}
- 平均绝对误差 (MAE): {metrics['mae']:.4f}
- 均方根误差 (RMSE): {metrics['rmse']:.4f}
- 解释方差分数 (EVS): {metrics['evs']:.4f}

训练信息:
- 最佳轮次: {training_history['best_epoch']}
- 最佳验证损失: {training_history['best_val_loss']:.4f}
- 总训练轮次: {training_history['total_epochs']}
"""
        
        return report
