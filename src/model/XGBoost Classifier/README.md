# XGBoost 分类器

预测某国在某年是否能获得奖牌（二分类：0或1）

## 运行

```bash
cd "E:\OneDrive\桌面\MCM_C\src\model\XGBoost Classifier"
python build_xgboost_classifier.py
```

## 输出文件

- **模型**: `xgboost_model.pkl` (当前目录)
- **预测结果**: `../../out/xgboost_predictions.csv`
- **特征重要性**: `../../out/feature_importance.csv`
- **特征列表**: `../../out/feature_list.txt`
- **可视化图表**: `../../figure/xgboost_model_analysis.png`
- **最佳超参数**: `../../out/best_hyperparameters.json`
- **搜索结果**: `../../out/hyperparameter_search_results.csv`

## 超参数搜索

模型使用 **RandomizedSearchCV** 自动搜索最佳超参数：

- **搜索方法**: RandomizedSearchCV（随机搜索50组参数）
- **交叉验证**: 5折KFold
- **评分标准**: AUC-ROC
- **搜索参数**:
  - `max_depth`: [3, 4, 5, 6, 7]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.15]
  - `n_estimators`: [100, 200, 300, 400]
  - `subsample`: [0.7, 0.8, 0.9]
  - `colsample_bytree`: [0.7, 0.8, 0.9]
  - `reg_alpha`: [0, 0.1, 0.5, 1.0]
  - `reg_lambda`: [0.5, 1.0, 1.5, 2.0]
  - `min_child_weight`: [1, 3, 5, 7]
  - `gamma`: [0, 0.1, 0.2, 0.3]

**输出文件**:
- `out/best_hyperparameters.json` - 最佳超参数
- `out/hyperparameter_search_results.csv` - 所有搜索结果

## 特征

使用30+个特征，包括：
- 历史表现特征（lag特征）
- 参赛规模特征
- 时间特征
- 国家特征
- 交互特征

详见 `out/feature_importance.csv`
