"""
配置管理模块
提供配置文件的加载、验证和管理功能
"""

import json
import os
from typing import Dict, Any, Optional, List
import logging


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")
    
    def _validate_config(self):
        """验证配置文件"""
        
        required_sections = [
            'data_paths', 'feature_columns', 'model_config', 
            'output_dirs', 'composition_constraints'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必需部分: {section}")
        
        # 验证数据路径
        for data_type, path in self.config['data_paths'].items():
            if not os.path.exists(path):
                logging.warning(f"数据文件不存在: {path}")
        
        # 验证输出目录
        for dir_name, dir_path in self.config['output_dirs'].items():
            os.makedirs(dir_path, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志"""
        
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 配置根日志器
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[]
        )
        
        # 添加控制台处理器
        if log_config.get('console_logging', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(format_str))
            logging.getLogger().addHandler(console_handler)
        
        # 添加文件处理器
        if log_config.get('file_logging', True):
            log_file = os.path.join(self.config['output_dirs']['logs'], 'pipeline.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(format_str))
            logging.getLogger().addHandler(file_handler)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_path(self, data_type: str) -> str:
        """获取数据路径"""
        
        path = self.get(f'data_paths.{data_type}')
        if not path:
            raise ValueError(f"未找到数据类型: {data_type}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")
        
        return path
    
    def get_feature_columns(self, feature_type: str) -> List[str]:
        """获取特征列名"""
        
        columns = self.get(f'feature_columns.{feature_type}')
        if not columns:
            raise ValueError(f"未找到特征类型: {feature_type}")
        
        return columns
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取模型配置"""
        
        config = self.get(f'model_config.{model_type}')
        if not config:
            raise ValueError(f"未找到模型类型: {model_type}")
        
        return config
    
    def get_output_dir(self, dir_type: str) -> str:
        """获取输出目录"""
        
        dir_path = self.get(f'output_dirs.{dir_type}')
        if not dir_path:
            raise ValueError(f"未找到输出目录类型: {dir_type}")
        
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    def get_composition_constraints(self) -> Dict[str, List[float]]:
        """获取成分约束"""
        
        constraints = self.get('composition_constraints')
        if not constraints:
            raise ValueError("未找到成分约束配置")
        
        return constraints
    
    def get_process_parameters(self, process_type: str) -> Dict[str, Any]:
        """获取工艺参数"""
        
        params = self.get(f'process_parameters.{process_type}')
        if not params:
            raise ValueError(f"未找到工艺参数类型: {process_type}")
        
        return params
    
    def update_config(self, key: str, value: Any):
        """更新配置值"""
        
        keys = key.split('.')
        config_ref = self.config
        
        # 导航到目标位置
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # 设置值
        config_ref[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None):
        """保存配置文件"""
        
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logging.info(f"配置文件已保存到: {output_path}")
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
            raise
    
    def create_experiment_config(self, experiment_name: str, 
                                overrides: Optional[Dict[str, Any]] = None) -> 'ConfigManager':
        """创建实验配置"""
        
        # 复制当前配置
        experiment_config = self.config.copy()
        
        # 更新实验信息
        experiment_config['experiment']['name'] = experiment_name
        
        # 应用覆盖配置
        if overrides:
            for key, value in overrides.items():
                self._update_nested_dict(experiment_config, key, value)
        
        # 创建临时配置文件
        temp_config_path = f"config_{experiment_name}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, indent=2, ensure_ascii=False)
        
        # 返回新的配置管理器
        return ConfigManager(temp_config_path)
    
    def _update_nested_dict(self, d: Dict[str, Any], key: str, value: Any):
        """更新嵌套字典"""
        
        keys = key.split('.')
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """获取可视化配置"""
        
        return self.get('visualization', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        
        return self.get('model_config.training', {})
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """获取优化配置"""
        
        return self.get('optimization', {})
    
    def is_feature_enabled(self, feature: str) -> bool:
        """检查功能是否启用"""
        
        return self.get(feature, {}).get('enabled', False)
    
    def get_device(self) -> str:
        """获取设备配置"""
        
        device = self.get('device', 'auto')
        if device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def get_random_state(self) -> int:
        """获取随机种子"""
        
        return self.get('random_state', 42)
    
    def validate_data_paths(self) -> Dict[str, bool]:
        """验证所有数据路径"""
        
        results = {}
        for data_type, path in self.config['data_paths'].items():
            results[data_type] = os.path.exists(path)
        return results
    
    def get_missing_data_files(self) -> List[str]:
        """获取缺失的数据文件"""
        
        missing = []
        for data_type, path in self.config['data_paths'].items():
            if not os.path.exists(path):
                missing.append(f"{data_type}: {path}")
        return missing


def load_config(config_path: str = "config.json") -> ConfigManager:
    """加载配置的便捷函数"""
    
    return ConfigManager(config_path)


def create_default_config(output_path: str = "config_default.json"):
    """创建默认配置文件"""
    
    default_config = {
        "random_state": 42,
        "device": "auto",
        "data_paths": {
            "vam_data": "data/VAM_LWRHEAS_property_AlNbTiVZrCrMoHfTaW_20241202-clean-RT.csv",
            "am_data": "data/AM_LWRHEAS_property_20231213-RT_train.csv",
            "am_valid": "data/AM_LWRHEAS_property_20231213-RT_valid.csv"
        },
        "feature_columns": {
            "composition": ["x(Al)", "x(Nb)", "x(Ti)", "x(V)", "x(Zr)", "x(Cr)", "x(Mo)", "x(Hf)", "x(Ta)"],
            "vam_process": ["anneal_temperature", "anneal_time", "C(0)/T(1)"],
            "am_process": ["LED", "AED", "VED", "C(0)/T(1)"],
            "parameters": ["Tm", "Delta-r", "Delta-X_Allen", "G", "Delta-G", "K", "DK", "W"],
            "properties": ["YS"]
        },
        "model_config": {
            "vam_net": {"input_size": 11, "hidden_sizes": [128, 64]},
            "am_feature_net": {"input_size": 11, "hidden_sizes": [256, 64, 32], "output_size": 10},
            "training": {"epochs": 1000, "learning_rate": 0.001, "weight_decay": 0.001, "patience": 50}
        },
        "output_dirs": {
            "models": "models",
            "results": "results", 
            "plots": "plots",
            "reports": "reports"
        },
        "composition_constraints": {
            "x(Al)": [0, 20], "x(Nb)": [0, 35], "x(Ti)": [0, 40],
            "x(V)": [0, 40], "x(Zr)": [0, 35], "x(Cr)": [0, 20],
            "x(Mo)": [0, 20], "x(Hf)": [0, 15], "x(Ta)": [0, 15]
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print(f"默认配置文件已创建: {output_path}")


if __name__ == '__main__':
    # 测试配置管理器
    try:
        config = ConfigManager()
        print("配置加载成功!")
        print(f"随机种子: {config.get_random_state()}")
        print(f"设备: {config.get_device()}")
        print(f"数据路径验证: {config.validate_data_paths()}")
    except Exception as e:
        print(f"配置加载失败: {e}")
