"""
数据处理和特征工程模块
包含数据加载、预处理、特征选择等功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from typing import List, Tuple, Optional, Dict, Any
from parameter_cal import parameter_calculation


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scalers = {}
        self.feature_names = {}
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, **kwargs)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
    
    def calculate_parameters(self, composition_data: pd.DataFrame, 
                           element_count: int) -> pd.DataFrame:
        """计算合金参数"""
        return parameter_calculation(composition_data, element_count)
    
    def prepare_vam_features(self, data: pd.DataFrame, 
                           composition_cols: List[str],
                           process_cols: List[str],
                           property_cols: List[str],
                           parameter_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """准备VAM特征数据"""
        
        # 计算参数
        df_parameter = self.calculate_parameters(data[composition_cols], len(composition_cols))
        
        # 合并所有特征
        df_features = pd.concat([
            data[composition_cols], 
            df_parameter[parameter_cols], 
            data[process_cols + property_cols]
        ], axis=1)
        
        # 保存特征名称
        self.feature_names['vam'] = parameter_cols + process_cols
        
        # 分离特征和目标
        X = df_features[parameter_cols + process_cols].values
        y = df_features[property_cols].values
        
        return X, y
    
    def prepare_am_features(self, data: pd.DataFrame,
                          composition_cols: List[str],
                          process_cols: List[str],
                          property_cols: List[str],
                          parameter_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """准备AM特征数据"""
        
        # 计算参数
        df_parameter = self.calculate_parameters(data[composition_cols], len(composition_cols))
        
        # 合并所有特征
        df_features = pd.concat([
            data[composition_cols],
            df_parameter[parameter_cols],
            data[process_cols + property_cols]
        ], axis=1)
        
        # 保存特征名称
        self.feature_names['am'] = parameter_cols + process_cols
        
        # 分离特征和目标
        X = df_features[parameter_cols + process_cols].values
        y = df_features[property_cols].values
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, **kwargs) -> Tuple[np.ndarray, ...]:
        """分割数据"""
        return train_test_split(X, y, test_size=test_size, 
                               random_state=self.random_state, **kwargs)
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray, 
                      scaler_type: str = 'standard', 
                      scaler_name: str = 'default') -> Tuple[np.ndarray, np.ndarray]:
        """特征标准化"""
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的标准化类型: {scaler_type}")
        
        # 拟合并转换训练数据
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 保存标准化器
        self.scalers[scaler_name] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, scaler_name: str, file_path: str):
        """保存标准化器"""
        if scaler_name not in self.scalers:
            raise ValueError(f"标准化器 {scaler_name} 不存在")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.scalers[scaler_name], file_path)
    
    def load_scaler(self, scaler_name: str, file_path: str):
        """加载标准化器"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"标准化器文件不存在: {file_path}")
        
        self.scalers[scaler_name] = joblib.load(file_path)
    
    def transform_features(self, X: np.ndarray, scaler_name: str) -> np.ndarray:
        """使用已保存的标准化器转换特征"""
        if scaler_name not in self.scalers:
            raise ValueError(f"标准化器 {scaler_name} 不存在")
        
        return self.scalers[scaler_name].transform(X)


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features = {}
    
    def variance_threshold_selection(self, X: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """方差阈值特征选择"""
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        # 保存选择的特征
        self.selected_features['variance'] = selector.get_support(indices=True)
        
        return X_selected
    
    def random_forest_selection(self, X: np.ndarray, y: np.ndarray, 
                              n_features: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """随机森林特征重要性选择"""
        
        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        
        # 获取特征重要性
        feature_importances = rf.feature_importances_
        
        if n_features is None:
            # 选择重要性大于平均值的特征
            threshold = np.mean(feature_importances)
            selected_indices = np.where(feature_importances > threshold)[0]
        else:
            # 选择前n个最重要的特征
            selected_indices = np.argsort(feature_importances)[-n_features:]
        
        X_selected = X[:, selected_indices]
        
        # 保存选择的特征
        self.selected_features['random_forest'] = selected_indices
        
        return X_selected, feature_importances
    
    def get_feature_importance_df(self, feature_names: List[str], 
                                importances: np.ndarray) -> pd.DataFrame:
        """获取特征重要性DataFrame"""
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)


class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_composition_distribution(self, data: pd.DataFrame, 
                                       composition_cols: List[str]) -> Dict[str, Any]:
        """分析成分分布"""
        
        # 将0值替换为NaN
        composition_data = data[composition_cols].replace(0, np.nan)
        
        # 统计非NaN值数量
        composition_data['count'] = composition_data.count(axis=1)
        
        # 元素统计
        element_counts = {}
        for col in composition_cols:
            element_counts[col] = composition_data[col].notna().sum()
        
        # 组合统计
        count_distribution = composition_data['count'].value_counts()
        
        results = {
            'element_counts': element_counts,
            'count_distribution': count_distribution,
            'composition_data': composition_data
        }
        
        self.analysis_results['composition'] = results
        return results
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, 
                                   columns: List[str]) -> pd.DataFrame:
        """计算相关性矩阵"""
        return data[columns].corr(method='spearman')
    
    def generate_summary_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成汇总统计"""
        return data.describe()
