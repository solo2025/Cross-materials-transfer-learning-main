"""
配置测试脚本
测试配置管理器的功能
"""

import os
import sys

# 添加core模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.config_manager import ConfigManager, create_default_config


def test_config_manager():
    """测试配置管理器"""
    
    print("=== 配置管理器测试 ===")
    
    try:
        # 测试配置加载
        config = ConfigManager("config.json")
        print("✅ 配置加载成功")
        
        # 测试基本功能
        print(f"随机种子: {config.get_random_state()}")
        print(f"设备: {config.get_device()}")
        
        # 测试数据路径验证
        data_paths = config.validate_data_paths()
        print("数据路径验证:")
        for data_type, exists in data_paths.items():
            status = "✅" if exists else "❌"
            print(f"  {status} {data_type}")
        
        # 测试缺失文件
        missing_files = config.get_missing_data_files()
        if missing_files:
            print("缺失的数据文件:")
            for file in missing_files:
                print(f"  ❌ {file}")
        else:
            print("✅ 所有数据文件都存在")
        
        # 测试特征列获取
        try:
            composition_cols = config.get_feature_columns('composition')
            print(f"✅ 成分列: {len(composition_cols)} 个")
            
            parameters = config.get_feature_columns('parameters')
            print(f"✅ 参数列: {len(parameters)} 个")
        except Exception as e:
            print(f"❌ 特征列获取失败: {e}")
        
        # 测试模型配置
        try:
            vam_config = config.get_model_config('vam_net')
            print(f"✅ VAM模型配置: {vam_config}")
        except Exception as e:
            print(f"❌ 模型配置获取失败: {e}")
        
        # 测试输出目录
        try:
            models_dir = config.get_output_dir('models')
            print(f"✅ 模型目录: {models_dir}")
        except Exception as e:
            print(f"❌ 输出目录获取失败: {e}")
        
        # 测试成分约束
        try:
            constraints = config.get_composition_constraints()
            print(f"✅ 成分约束: {len(constraints)} 个元素")
        except Exception as e:
            print(f"❌ 成分约束获取失败: {e}")
        
        print("\n=== 配置测试完成 ===")
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")


def test_config_creation():
    """测试配置创建"""
    
    print("\n=== 配置创建测试 ===")
    
    try:
        # 创建默认配置
        create_default_config("config_test.json")
        print("✅ 默认配置文件创建成功")
        
        # 测试加载创建的配置
        test_config = ConfigManager("config_test.json")
        print("✅ 创建的配置文件加载成功")
        
        # 清理测试文件
        os.remove("config_test.json")
        print("✅ 测试文件清理完成")
        
    except Exception as e:
        print(f"❌ 配置创建测试失败: {e}")


def test_config_updates():
    """测试配置更新"""
    
    print("\n=== 配置更新测试 ===")
    
    try:
        config = ConfigManager("config.json")
        
        # 测试配置更新
        original_value = config.get('random_state')
        config.update_config('random_state', 123)
        updated_value = config.get('random_state')
        
        if updated_value == 123:
            print("✅ 配置更新成功")
        else:
            print("❌ 配置更新失败")
        
        # 恢复原始值
        config.update_config('random_state', original_value)
        print("✅ 配置恢复成功")
        
    except Exception as e:
        print(f"❌ 配置更新测试失败: {e}")


def main():
    """主函数"""
    
    print("配置管理器功能测试")
    print("=" * 50)
    
    # 运行测试
    test_config_manager()
    test_config_creation()
    test_config_updates()
    
    print("\n所有测试完成!")


if __name__ == '__main__':
    main()
