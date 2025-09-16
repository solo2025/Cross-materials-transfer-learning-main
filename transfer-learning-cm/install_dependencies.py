"""
依赖包安装脚本
确保所有必需的包都能正确安装
"""

import subprocess
import sys

def install_package(package):
    """安装单个包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ {package} 安装失败")
        return False

def check_package(package):
    """检查包是否已安装"""
    try:
        __import__(package)
        print(f"✓ {package} 已安装")
        return True
    except ImportError:
        print(f"✗ {package} 未安装")
        return False

def main():
    """主函数"""
    print("检查并安装依赖包...")
    print("=" * 50)
    
    # 需要安装的包列表
    packages = [
        "numpy",
        "pandas", 
        "scikit-learn",
        "matplotlib",
        "xgboost",
        "shap"
    ]
    
    # 检查已安装的包
    installed_packages = []
    missing_packages = []
    
    for package in packages:
        if check_package(package):
            installed_packages.append(package)
        else:
            missing_packages.append(package)
    
    print(f"\n已安装: {len(installed_packages)} 个包")
    print(f"需要安装: {len(missing_packages)} 个包")
    
    # 安装缺失的包
    if missing_packages:
        print(f"\n开始安装缺失的包...")
        for package in missing_packages:
            install_package(package)
    
    # 特殊处理keras和tensorflow
    print(f"\n检查深度学习框架...")
    try:
        import tensorflow as tf
        print(f"✓ tensorflow 已安装 (版本: {tf.__version__})")
    except ImportError:
        print("✗ tensorflow 未安装，正在安装...")
        install_package("tensorflow")
    
    try:
        import keras
        print(f"✓ keras 已安装 (版本: {keras.__version__})")
    except ImportError:
        print("✗ keras 未安装，正在安装...")
        install_package("keras")
    
    print("\n" + "=" * 50)
    print("依赖包检查完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
