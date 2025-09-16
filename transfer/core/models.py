"""
统一的神经网络模型定义模块
包含VAM和AM模型的所有网络结构
"""

import torch
import torch.nn as nn
from typing import Optional


class VAMNet(nn.Module):
    """VAM数据预测模型"""
    
    def __init__(self, input_size: int = 11, hidden_sizes: list = [128, 64]):
        super(VAMNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class AMFeatureNet(nn.Module):
    """AM特征提取网络"""
    
    def __init__(self, input_size: int = 11, hidden_sizes: list = [256, 64, 32], output_size: int = 10):
        super(AMFeatureNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TransferLearningNet(nn.Module):
    """迁移学习组合网络"""
    
    def __init__(self, feature_net: AMFeatureNet, property_net: VAMNet):
        super(TransferLearningNet, self).__init__()
        self.feature_net = feature_net
        self.property_net = property_net
        
        # 冻结属性网络参数
        for param in self.property_net.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 提取特征
        features = self.feature_net(x[:, :-1])  # 除了最后一个条件参数
        # 组合特征和条件
        combined = torch.cat([features, x[:, -1:]], dim=1)
        # 预测属性
        return self.property_net(combined)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.models = {}
    
    def load_model(self, model_name: str, model_path: str, model_class: nn.Module, **kwargs):
        """加载预训练模型"""
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.models[model_name] = model
        return model
    
    def get_model(self, model_name: str):
        """获取已加载的模型"""
        return self.models.get(model_name)
    
    def predict(self, model_name: str, data: torch.Tensor):
        """使用指定模型进行预测"""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"模型 {model_name} 未找到")
        
        with torch.no_grad():
            return model(data)
