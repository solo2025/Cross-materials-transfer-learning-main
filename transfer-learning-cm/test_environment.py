"""
环境测试脚本
测试所有依赖包是否正确安装
"""

def test_imports():
    """测试所有必需的导入"""
    print("测试依赖包导入...")
    print("=" * 40)
    
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'xgboost': 'xgboost',
        'shap': 'shap',
        'tensorflow': 'tensorflow',
        'keras': 'keras'
    }
    
    success_count = 0
    total_count = len(packages)
    
    for name, module in packages.items():
        try:
            __import__(module)
            print(f"✓ {name}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {name}: {e}")
    
    print("=" * 40)
    print(f"成功导入: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 所有依赖包都已正确安装!")
        return True
    else:
        print("⚠️  部分依赖包缺失，请运行以下命令安装:")
        print("pip install numpy pandas scikit-learn matplotlib xgboost shap tensorflow keras")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    print("=" * 40)
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        
        # 测试numpy
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✓ numpy 基本功能正常: {arr.mean()}")
        
        # 测试pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"✓ pandas 基本功能正常: {df.shape}")
        
        # 测试sklearn
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform([[1], [2], [3]])
        print(f"✓ sklearn 基本功能正常: {scaled.shape}")
        
        print("✓ 基本功能测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("跨材料迁移学习项目 - 环境测试")
    print("=" * 50)
    
    # 测试导入
    import_success = test_imports()
    
    # 测试基本功能
    if import_success:
        functionality_success = test_basic_functionality()
        
        if functionality_success:
            print("\n🎉 环境测试完全通过!")
            print("您可以运行以下命令开始使用:")
            print("python run_refactored.py")
        else:
            print("\n⚠️  环境测试部分失败")
    else:
        print("\n❌ 环境测试失败")

if __name__ == "__main__":
    main()
