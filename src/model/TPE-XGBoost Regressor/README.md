# TPE-XGBoost 回归器

对分类器预测为1的样本（获得奖牌的国家），预测具体的奖牌数量。

## 特点

- **目标**: 预测获得奖牌国家的具体奖牌数量（Total）
- **TPE超参数搜索**: 使用 hyperopt 库的 Tree-structured Parzen Estimator
- **时间序列交叉验证**: Last-block Cross-Validation
  - CV1: 训练(1960-2012) -> 预测(2016)
  - CV2: 训练(1960-2016) -> 预测(2024)
- **Bootstrap置信区间**: 1000次重采样，计算95%置信区间
- **损失函数**: MSE (Mean Squared Error)

## 运行

```bash
cd "E:\OneDrive\桌面\MCM_C\src\model\TPE-XGBoost Regressor"
python build_xgboost_regressor.py
```

**注意**: 需要安装 `hyperopt` 库：
```bash
pip install hyperopt
```

## 输出文件

- **模型**: `xgboost_regressor_tpe.pkl` (当前目录)
- **预测结果+置信区间**: `../../out/xgboost_regressor_predictions.csv`
- **Bootstrap统计**: `../../out/bootstrap_regressor_confidence_intervals.csv`
- **TPE最佳参数**: `../../out/best_hyperparameters_regressor_tpe.json`
- **特征重要性**: `../../out/regressor_feature_importance.csv`
- **可视化图表**: `../../figure/xgboost_regressor_analysis.png`

## 超参数搜索

- **搜索方法**: TPE (Tree-structured Parzen Estimator) - hyperopt库
- **优化算法**: 贝叶斯优化，智能探索参数空间
- **损失函数**: MSE (Mean Squared Error)
- **迭代次数**: 50次
- **搜索参数范围**:
  - `max_depth`: [3, 10] (整数)
  - `learning_rate`: [0.01, 0.3] (对数均匀分布)
  - `n_estimators`: [100, 500] (整数，步长50)
  - `subsample`: [0.6, 1.0] (均匀分布)
  - `colsample_bytree`: [0.6, 1.0] (均匀分布)
  - `reg_alpha`: [0, 2.0] (均匀分布)
  - `reg_lambda`: [0.5, 3.0] (均匀分布)
  - `min_child_weight`: [1, 10] (整数)
  - `gamma`: [0, 0.5] (均匀分布)

## 数据说明

- **训练数据**: 只使用 `has_medal == 1` 的样本（获得奖牌的国家）
- **预测目标**: `Total` (总奖牌数)
- **时间范围**: 1960-2024
- **特征**: 与分类器相同的30+个特征

## 评估指标

- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **Bootstrap置信区间**: 95%置信区间

## 工作流程

1. **分类器**: 预测国家是否能获得奖牌（0或1）
2. **回归器**: 对分类器预测为1的样本，预测具体奖牌数量

两个模型配合使用，实现完整的预测流程。
