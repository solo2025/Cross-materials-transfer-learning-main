@echo off
echo 安装跨材料迁移学习项目依赖包...
echo ========================================

echo 安装基础包...
pip install numpy pandas scikit-learn matplotlib

echo 安装机器学习包...
pip install xgboost shap

echo 安装深度学习包...
pip install tensorflow keras

echo ========================================
echo 安装完成！
echo 运行 python test_environment.py 测试环境
pause
