"""
运行脚本 - 演示如何使用重构后的代码
"""

import os
import sys

def check_dependencies():
    """检查依赖库是否安装"""
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'keras', 
        'tensorflow', 'matplotlib', 'xgboost', 'shap'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("缺少以下依赖库:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    print("所有依赖库已安装!")
    return True

def check_data_files():
    """检查数据文件是否存在"""
    required_files = [
        'Superalloys.csv',
        'RT<=100.csv',
        'RT>100h, 35 points.csv',
        'eliminate 38 points.csv',
        'T<=600, S>=300, 38 points.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("缺少以下数据文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请确保所有数据文件都在当前目录中")
        return False
    
    print("所有数据文件都存在!")
    return True

def run_source_cnn():
    """运行源CNN模型训练"""
    print("\n" + "="*50)
    print("运行源CNN模型训练")
    print("="*50)
    
    try:
        from source_cnn_refactored import main as source_cnn_main
        source_cnn_main()
        print("源CNN模型训练完成!")
        return True
    except Exception as e:
        print(f"源CNN模型训练失败: {e}")
        return False

def run_transfer_cnn():
    """运行迁移学习CNN模型训练"""
    print("\n" + "="*50)
    print("运行迁移学习CNN模型训练")
    print("="*50)
    
    try:
        from transfer_cnn_refactored import main as transfer_cnn_main
        transfer_cnn_main()
        print("迁移学习CNN模型训练完成!")
        return True
    except Exception as e:
        print(f"迁移学习CNN模型训练失败: {e}")
        return False

def run_data_fusion():
    """运行数据融合迁移学习"""
    print("\n" + "="*50)
    print("运行数据融合迁移学习")
    print("="*50)
    
    try:
        from data_fusion_refactored import main as data_fusion_main
        data_fusion_main()
        print("数据融合迁移学习完成!")
        return True
    except Exception as e:
        print(f"数据融合迁移学习失败: {e}")
        return False

def main():
    """主函数"""
    print("跨材料迁移学习项目 - 重构版")
    print("="*50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查数据文件
    if not check_data_files():
        return
    
    print("\n选择要运行的模块:")
    print("1. 源CNN模型训练")
    print("2. 迁移学习CNN模型训练")
    print("3. 数据融合迁移学习")
    print("4. 运行所有模块")
    print("0. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (0-4): ").strip()
            
            if choice == '0':
                print("退出程序")
                break
            elif choice == '1':
                run_source_cnn()
            elif choice == '2':
                run_transfer_cnn()
            elif choice == '3':
                run_data_fusion()
            elif choice == '4':
                print("运行所有模块...")
                run_source_cnn()
                run_transfer_cnn()
                run_data_fusion()
                print("\n所有模块运行完成!")
                break
            else:
                print("无效选择，请重新输入")
                continue
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"运行出错: {e}")
            break

if __name__ == "__main__":
    main()
