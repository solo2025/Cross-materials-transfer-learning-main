"""
配置模块 - 统一管理所有配置参数
"""

# 数据文件路径配置
DATA_CONFIG = {
    'source_cnn': {
        'data_file': 'Superalloys.csv',
        'target_column': 'class',
        'feature_columns': None,  # 使用除target_column外的所有列
        'test_size': 0.2,
        'random_state': 42
    },
    'transfer_cnn': {
        'train_file': 'RT<=100.csv',
        'test_file': 'RT>100h, 35 points.csv',
        'target_column': 'Ti_lg_RT',
        'test_size': 0.2,
        'random_state': 42
    },
    'data_fusion': {
        'train_file1': 'Superalloys.csv',
        'train_file2': 'eliminate 38 points.csv',
        'test_file': 'T<=600, S>=300, 38 points.csv',
        'target_column': 'lg_RT',
        'test_size': 0.2,
        'random_state': 42
    }
}

# 模型配置
MODEL_CONFIG = {
    'cnn': {
        'input_shape': (6, 6, 1),
        'conv_filters': [8, 16],
        'conv_kernel_size': (2, 2),
        'dense_units': [128],
        'output_units': 1,
        'activation': 'relu',
        'padding': 'same',
        'optimizer': 'adam',
        'learning_rate': 0.005,
        'loss': 'mse',
        'epochs': 1500,
        'batch_size': 50,
        'validation_split': 0.2
    },
    'ml_models': {
        'random_forest': {
            'n_estimators': [10, 20, 30],
            'max_depth': [15, 16, 17],
            'min_samples_split': [4, 5, 6],
            'min_samples_leaf': [3, 4, 5]
        },
        'gaussian_process': {
            'normalize_y': [False],
            'kernel': ['None', 'DotProduct', 'DotProduct+WhiteKernel', 'WhiteKernel'],
            'alpha': 'np.arange(0.001, 0.1, 0.001)'
        },
        'svr': {
            'C': [2, 3, 4, 5, 6, 7, 8],
            'epsilon': 'np.arange(0.01, 1, 0.01)'
        },
        'xgboost': {
            'n_estimators': [160, 170],
            'max_depth': [5, 6, 7],
            'min_child_weight': [2, 3, 4],
            'gamma': [0.001, 0.01, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_lambda': [2, 3, 5, 8],
            'reg_alpha': [0, 0.1]
        },
        'adaboost': {
            'n_estimators': [20, 30, 40, 50, 60, 80, 100]
        }
    }
}

# 可视化配置
PLOT_CONFIG = {
    'figure_size': (6, 6),
    'dpi': 500,
    'font_family': 'Times New Roman',
    'font_size': 14,
    'tick_size': 12,
    'colors': {
        'train1': 'aquamarine',
        'train2': 'teal',
        'test': 'navy',
        'line': 'grey'
    },
    'markers': {
        'train1': 'D',
        'train2': '>',
        'test': '*'
    }
}

# 评估指标配置
EVALUATION_CONFIG = {
    'metrics': ['mse', 'rmse', 'mae', 'r2', 'mape'],
    'cv_folds': 5,
    'test_runs': 1
}
