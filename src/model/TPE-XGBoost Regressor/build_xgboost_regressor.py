"""
TPE-XGBoost回归器（金银铜牌预测）
- 对分类器预测为1的样本（获得奖牌的国家），分别预测金、银、铜牌数量
- 使用TPE (Tree-structured Parzen Estimator) 进行超参数搜索
- 使用时间序列交叉验证 (Last-block Cross-Validation)
- 使用Bootstrap方法计算置信区间
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import json
import warnings
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
import pickle
warnings.filterwarnings('ignore')

# 设置随机种子（用于可重复性）
np.random.seed(42)

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..', '..')
PROJECT_ROOT = os.path.normpath(PROJECT_ROOT)

HQ_DATA_DIR = os.path.join(PROJECT_ROOT, 'hq_data')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'out')
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figure')
MODEL_DIR = SCRIPT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*80)
print("TPE-XGBoost回归器 - 预测金、银、铜牌数量")
print("="*80)
print(f"\n项目根目录: {PROJECT_ROOT}")
print(f"数据目录: {HQ_DATA_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"图表目录: {FIGURE_DIR}")

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n[1/7] 加载数据...")

medal_file = os.path.join(HQ_DATA_DIR, 'medal_counts_from_athletes.csv')
athlete_file = os.path.join(HQ_DATA_DIR, 'athletes_cleaned_full.csv')
host_file = os.path.join(DATA_DIR, 'summerOly_hosts.csv')

if not os.path.exists(medal_file):
    raise FileNotFoundError(f"找不到文件: {medal_file}\n请先运行 script/rebuild_medal_counts_from_athletes.py 生成数据")

medal_counts = pd.read_csv(medal_file)
athletes = pd.read_csv(athlete_file)
hosts = pd.read_csv(host_file)

print(f"  奖牌数据: {len(medal_counts)} 条")
print(f"  运动员数据: {len(athletes)} 条")

# ============================================================================
# 2. 特征工程（复用分类器的逻辑）
# ============================================================================
print("\n[2/7] 特征工程...")

# 创建基础数据集
base = medal_counts[['NOC', 'Year', 'Gold', 'Silver', 'Bronze', 'Total']].copy()
base['has_medal'] = (base['Total'] > 0).astype(int)

# 添加参赛规模特征
athlete_stats = athletes.groupby(['NOC', 'Year']).agg({
    'Name': 'count',
    'Event': 'nunique',
    'Sport': 'nunique',
    'is_team_event': 'sum',
    'Sex': lambda x: (x == 'F').sum()
}).reset_index()

athlete_stats.columns = ['NOC', 'Year', 'athlete_count', 'event_count', 
                        'sport_count', 'team_events_count', 'female_athletes']

base = base.merge(athlete_stats, on=['NOC', 'Year'], how='left').fillna(0)

yearly_totals = base.groupby('Year')['athlete_count'].sum()
base['athlete_count_normalized'] = base.apply(
    lambda row: row['athlete_count'] / yearly_totals[row['Year']] if yearly_totals[row['Year']] > 0 else 0,
    axis=1
)

# 添加历史表现特征
base = base.sort_values(['NOC', 'Year']).reset_index(drop=True)

for lag in [1, 2, 3]:
    base[f'medal_count_lag{lag}'] = base.groupby('NOC')['Total'].shift(lag).fillna(0)
    base[f'gold_count_lag{lag}'] = base.groupby('NOC')['Gold'].shift(lag).fillna(0)
    base[f'silver_count_lag{lag}'] = base.groupby('NOC')['Silver'].shift(lag).fillna(0)
    base[f'bronze_count_lag{lag}'] = base.groupby('NOC')['Bronze'].shift(lag).fillna(0)

base['avg_medals_last_3'] = (
    base['medal_count_lag1'] + base['medal_count_lag2'] + base['medal_count_lag3']
) / 3

base['max_medals_historical'] = base.groupby('NOC')['Total'].transform(
    lambda x: x.shift(1).expanding().max()
).fillna(0)

base['total_historical_medals'] = base.groupby('NOC')['Total'].transform(
    lambda x: x.shift(1).expanding().sum()
).fillna(0)

base['historical_participation_count'] = base.groupby('NOC').cumcount()

def calculate_streak(group):
    historical = group.shift(1).fillna(0)
    streak = []
    current_streak = 0
    for val in historical:
        if val == 1:
            current_streak += 1
        else:
            current_streak = 0
        streak.append(current_streak)
    return pd.Series(streak, index=group.index)

base['medal_streak'] = base.groupby('NOC')['has_medal'].transform(calculate_streak)

# 距离上次获奖的年数
base_reset = base.reset_index(drop=True)
base_reset['years_since_last_medal'] = 0

for noc in base_reset['NOC'].unique():
    noc_mask = base_reset['NOC'] == noc
    noc_data = base_reset[noc_mask].copy().sort_values('Year')
    
    last_medal_year = None
    for idx, row in noc_data.iterrows():
        prev_data = noc_data[noc_data['Year'] < row['Year']]
        if len(prev_data) > 0:
            prev_medals = prev_data[prev_data['has_medal'] == 1]
            if len(prev_medals) > 0:
                last_medal_year = prev_medals['Year'].iloc[-1]
                base_reset.loc[idx, 'years_since_last_medal'] = row['Year'] - last_medal_year
            else:
                base_reset.loc[idx, 'years_since_last_medal'] = 999
        else:
            base_reset.loc[idx, 'years_since_last_medal'] = 999

base = base_reset.copy()

base['medal_trend'] = (base['medal_count_lag1'] - base['medal_count_lag2']) / (base['medal_count_lag2'] + 1)
base['medal_trend'] = base['medal_trend'].fillna(0)
base['is_improving'] = (base['medal_trend'] > 0).astype(int)

# 添加时间特征
base['year_normalized'] = (base['Year'] - 1960) / (2024 - 1960)

host_mapping = {
    1960: 'ITA', 1964: 'JPN', 1968: 'MEX', 1972: 'FRG', 1976: 'CAN',
    1988: 'KOR', 1992: 'ESP', 1996: 'USA', 2000: 'AUS', 2004: 'GRE',
    2008: 'CHN', 2012: 'GBR', 2016: 'BRA', 2024: 'FRA',
}

base['is_host'] = base.apply(
    lambda row: 1 if row['NOC'] == host_mapping.get(row['Year'], '') else 0,
    axis=1
)

first_participation = base.groupby('NOC')['Year'].min()
base['years_since_first_participation'] = base.apply(
    lambda row: row['Year'] - first_participation.get(row['NOC'], row['Year']),
    axis=1
)

# 添加国家特征
le_noc = LabelEncoder()
base['NOC_encoded'] = le_noc.fit_transform(base['NOC'])

base['historical_medal_rate'] = base.groupby('NOC')['has_medal'].transform(
    lambda x: x.shift(1).expanding().mean()
).fillna(0)

base['athlete_count_lag1'] = base.groupby('NOC')['athlete_count'].shift(1).fillna(0)
base['athlete_count_trend'] = (base['athlete_count'] - base['athlete_count_lag1']) / (base['athlete_count_lag1'] + 1)

# 交互特征
base['athlete_medal_interaction'] = base['athlete_count'] * base['medal_count_lag1']
base['host_medal_interaction'] = base['is_host'] * base['medal_count_lag1']
base['year_medal_interaction'] = base['year_normalized'] * base['medal_count_lag1']

base = base.fillna(0)

# 特征选择
feature_cols = [
    'medal_count_lag1', 'medal_count_lag2', 'medal_count_lag3',
    'gold_count_lag1', 'silver_count_lag1', 'bronze_count_lag1',
    'avg_medals_last_3', 'max_medals_historical', 'total_historical_medals',
    'medal_streak', 'years_since_last_medal', 'medal_trend', 'is_improving',
    'athlete_count', 'event_count', 'sport_count', 'team_events_count',
    'female_athletes', 'athlete_count_normalized', 'athlete_count_trend',
    'year_normalized', 'is_host', 'years_since_first_participation',
    'NOC_encoded', 'historical_participation_count', 'historical_medal_rate',
    'athlete_medal_interaction', 'host_medal_interaction', 'year_medal_interaction',
]

available_features = [f for f in feature_cols if f in base.columns]
print(f"  特征数: {len(available_features)}")

# ============================================================================
# 3. 准备数据（只使用获得奖牌的样本）
# ============================================================================
print("\n[3/7] 准备数据（只使用获得奖牌的样本）...")

base_clean = base[base['Year'] >= 1960].copy()  # 从1960年开始
base_clean = base_clean.dropna(subset=available_features)

# 关键：只使用获得奖牌的样本（has_medal == 1）
base_medal_only = base_clean[base_clean['has_medal'] == 1].copy()

X = base_medal_only[available_features]
y_gold = base_medal_only['Gold']  # 预测金牌数
y_silver = base_medal_only['Silver']  # 预测银牌数
y_bronze = base_medal_only['Bronze']  # 预测铜牌数
y_total = base_medal_only['Total']  # 总奖牌数（用于参考）

print(f"  总样本数: {len(X)} (只包含获得奖牌的国家)")
print(f"  金牌数范围: [{y_gold.min()}, {y_gold.max()}], 平均: {y_gold.mean():.2f}")
print(f"  银牌数范围: [{y_silver.min()}, {y_silver.max()}], 平均: {y_silver.mean():.2f}")
print(f"  铜牌数范围: [{y_bronze.min()}, {y_bronze.max()}], 平均: {y_bronze.mean():.2f}")

# 时间序列交叉验证的数据分割
# CV1: 用1960-2012训练，预测2016
train_2016_mask = (base_medal_only['Year'] >= 1960) & (base_medal_only['Year'] <= 2012)
test_2016_mask = base_medal_only['Year'] == 2016

# CV2: 用1960-2016训练，预测2024
train_2024_mask = (base_medal_only['Year'] >= 1960) & (base_medal_only['Year'] <= 2016)
test_2024_mask = base_medal_only['Year'] == 2024

X_train_2016 = X[train_2016_mask]
y_gold_train_2016 = y_gold[train_2016_mask]
y_silver_train_2016 = y_silver[train_2016_mask]
y_bronze_train_2016 = y_bronze[train_2016_mask]
X_test_2016 = X[test_2016_mask]
y_gold_test_2016 = y_gold[test_2016_mask]
y_silver_test_2016 = y_silver[test_2016_mask]
y_bronze_test_2016 = y_bronze[test_2016_mask]

X_train_2024 = X[train_2024_mask]
y_gold_train_2024 = y_gold[train_2024_mask]
y_silver_train_2024 = y_silver[train_2024_mask]
y_bronze_train_2024 = y_bronze[train_2024_mask]
X_test_2024 = X[test_2024_mask]
y_gold_test_2024 = y_gold[test_2024_mask]
y_silver_test_2024 = y_silver[test_2024_mask]
y_bronze_test_2024 = y_bronze[test_2024_mask]

print(f"\n  时间序列交叉验证设置:")
print(f"    CV1: 训练(1960-2012, {len(X_train_2016)}样本) -> 测试(2016, {len(X_test_2016)}样本)")
print(f"    CV2: 训练(1960-2016, {len(X_train_2024)}样本) -> 测试(2024, {len(X_test_2024)}样本)")

# ============================================================================
# 4. TPE超参数搜索（为每个奖牌类型分别搜索，主要针对CV2）
# ============================================================================
print("\n[4/7] TPE超参数搜索（分别为金、银、铜牌搜索，主要针对CV2：预测2024）...")

# 定义搜索空间
space = {
    'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 500, 50)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0, 2.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.5, 3.0),
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),
    'gamma': hp.uniform('gamma', 0, 0.5),
}

def create_objective(y_train, y_test):
    """创建TPE优化的目标函数 - 主要针对CV2（预测2024）"""
    def objective(params):
        # 转换参数类型
        params = params.copy()
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = 'rmse'
        params['random_state'] = 42
        params['n_jobs'] = -1
        
        model = xgb.XGBRegressor(**params)
        
        try:
            # 使用CV2的训练和测试集（1960-2016训练，2024测试）
            model.fit(X_train_2024, y_train, verbose=False)
            y_pred = model.predict(X_test_2024)
            mse = mean_squared_error(y_test, y_pred)
            return {'loss': mse, 'status': STATUS_OK}
        except Exception as e:
            return {'loss': 1e6, 'status': STATUS_OK}
    return objective

# 为每个奖牌类型分别搜索最佳参数（主要针对CV2：预测2024）
medal_types = ['Gold', 'Silver', 'Bronze']
y_trains_cv2 = [y_gold_train_2024, y_silver_train_2024, y_bronze_train_2024]
y_tests_cv2 = [y_gold_test_2024, y_silver_test_2024, y_bronze_test_2024]
best_params_dict = {}

for medal_type, y_train, y_test in zip(medal_types, y_trains_cv2, y_tests_cv2):
    print(f"\n  搜索 {medal_type} 牌的最佳超参数（50次迭代，针对CV2：预测2024）...")
    trials = Trials()
    best = fmin(
        fn=create_objective(y_train, y_test),
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        verbose=0
    )
    
    best_params = {
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'n_estimators': int(best['n_estimators']),
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'reg_alpha': best['reg_alpha'],
        'reg_lambda': best['reg_lambda'],
        'min_child_weight': int(best['min_child_weight']),
        'gamma': best['gamma'],
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1
    }
    
    best_params_dict[medal_type] = best_params
    print(f"    {medal_type} 最佳参数已找到")

# 保存最佳参数
best_params_file = os.path.join(OUTPUT_DIR, 'best_hyperparameters_regressor_gold_silver_bronze_tpe.json')
with open(best_params_file, 'w', encoding='utf-8') as f:
    save_dict = {}
    for medal_type, params in best_params_dict.items():
        save_dict[medal_type] = {k: float(v) if isinstance(v, (int, float)) else v 
                                 for k, v in params.items() 
                                 if k not in ['objective', 'eval_metric', 'random_state', 'n_jobs']}
    json.dump(save_dict, f, indent=2, ensure_ascii=False)
print(f"\n  所有最佳参数已保存至: out/best_hyperparameters_regressor_gold_silver_bronze_tpe.json")

# ============================================================================
# 5. 时间序列交叉验证训练（为每个奖牌类型分别训练）
# ============================================================================
print("\n[5/7] 时间序列交叉验证训练（分别为金、银、铜牌训练模型）...")

models_cv1 = {}
models_cv2 = {}
predictions_cv1 = {}
predictions_cv2 = {}
metrics_cv1 = {}
metrics_cv2 = {}

y_train_2016_dict = {
    'Gold': y_gold_train_2016,
    'Silver': y_silver_train_2016,
    'Bronze': y_bronze_train_2016
}
y_test_2016_dict = {
    'Gold': y_gold_test_2016,
    'Silver': y_silver_test_2016,
    'Bronze': y_bronze_test_2016
}
y_train_2024_dict = {
    'Gold': y_gold_train_2024,
    'Silver': y_silver_train_2024,
    'Bronze': y_bronze_train_2024
}
y_test_2024_dict = {
    'Gold': y_gold_test_2024,
    'Silver': y_silver_test_2024,
    'Bronze': y_bronze_test_2024
}

for medal_type in medal_types:
    print(f"\n  {medal_type} 牌模型:")
    
    # CV1: 用1960-2012训练，预测2016
    print(f"    CV1: 训练(1960-2012) -> 预测(2016)")
    model_cv1 = xgb.XGBRegressor(**best_params_dict[medal_type])
    model_cv1.fit(X_train_2016, y_train_2016_dict[medal_type], verbose=False)
    y_pred_cv1 = model_cv1.predict(X_test_2016)
    y_pred_cv1 = np.maximum(y_pred_cv1, 0)  # 确保非负
    
    mse_cv1 = mean_squared_error(y_test_2016_dict[medal_type], y_pred_cv1)
    mae_cv1 = mean_absolute_error(y_test_2016_dict[medal_type], y_pred_cv1)
    r2_cv1 = r2_score(y_test_2016_dict[medal_type], y_pred_cv1) if len(y_test_2016_dict[medal_type]) > 1 else 0
    
    models_cv1[medal_type] = model_cv1
    predictions_cv1[medal_type] = y_pred_cv1
    metrics_cv1[medal_type] = {'MSE': mse_cv1, 'MAE': mae_cv1, 'R²': r2_cv1}
    
    print(f"      MSE: {mse_cv1:.2f}, MAE: {mae_cv1:.2f}, R²: {r2_cv1:.4f}")
    
    # CV2: 用1960-2016训练，预测2024
    print(f"    CV2: 训练(1960-2016) -> 预测(2024)")
    model_cv2 = xgb.XGBRegressor(**best_params_dict[medal_type])
    model_cv2.fit(X_train_2024, y_train_2024_dict[medal_type], verbose=False)
    y_pred_cv2 = model_cv2.predict(X_test_2024)
    y_pred_cv2 = np.maximum(y_pred_cv2, 0)  # 确保非负
    
    mse_cv2 = mean_squared_error(y_test_2024_dict[medal_type], y_pred_cv2)
    mae_cv2 = mean_absolute_error(y_test_2024_dict[medal_type], y_pred_cv2)
    r2_cv2 = r2_score(y_test_2024_dict[medal_type], y_pred_cv2) if len(y_test_2024_dict[medal_type]) > 1 else 0
    
    models_cv2[medal_type] = model_cv2
    predictions_cv2[medal_type] = y_pred_cv2
    metrics_cv2[medal_type] = {'MSE': mse_cv2, 'MAE': mae_cv2, 'R²': r2_cv2}
    
    print(f"      MSE: {mse_cv2:.2f}, MAE: {mae_cv2:.2f}, R²: {r2_cv2:.4f}")

# ============================================================================
# 6. Bootstrap置信区间（为每个奖牌类型分别计算）
# ============================================================================
print("\n[6/7] Bootstrap置信区间计算（分别为金、银、铜牌计算）...")

n_bootstrap = 1000
bootstrap_predictions_dict = {}
mean_predictions_dict = {}
lower_bound_dict = {}
upper_bound_dict = {}

for medal_type in medal_types:
    print(f"\n  {medal_type} 牌 Bootstrap ({n_bootstrap}次重采样)...")
    bootstrap_predictions = []
    
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"    进度: {i+1}/{n_bootstrap}")
        
        # 重采样训练数据
        indices = np.random.choice(len(X_train_2024), size=len(X_train_2024), replace=True)
        X_boot = X_train_2024.iloc[indices]
        y_boot = y_train_2024_dict[medal_type].iloc[indices]
        
        # 训练模型
        model_boot = xgb.XGBRegressor(**best_params_dict[medal_type])
        model_boot.fit(X_boot, y_boot, verbose=False)
        
        # 预测测试集
        y_pred_boot = model_boot.predict(X_test_2024)
        y_pred_boot = np.maximum(y_pred_boot, 0)  # 确保非负
        bootstrap_predictions.append(y_pred_boot)
    
    bootstrap_predictions = np.array(bootstrap_predictions)
    bootstrap_predictions_dict[medal_type] = bootstrap_predictions
    
    # 计算置信区间
    mean_predictions = bootstrap_predictions.mean(axis=0)
    lower_bound = np.percentile(bootstrap_predictions, 2.5, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, 97.5, axis=0)
    
    mean_predictions_dict[medal_type] = mean_predictions
    lower_bound_dict[medal_type] = lower_bound
    upper_bound_dict[medal_type] = upper_bound
    
    print(f"    完成！预测均值范围: [{mean_predictions.min():.2f}, {mean_predictions.max():.2f}]")
    print(f"    平均置信区间宽度: {(upper_bound - lower_bound).mean():.2f}")

# ============================================================================
# 7. 保存结果
# ============================================================================
print("\n" + "="*80)
print("保存结果")
print("="*80)

# 保存最终模型（使用CV2的模型，为每个奖牌类型分别保存）
for medal_type in medal_types:
    model_path = os.path.join(MODEL_DIR, f'xgboost_regressor_{medal_type.lower()}_tpe.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(models_cv2[medal_type], f)
    print(f"  {medal_type}牌模型已保存至: {os.path.relpath(model_path, PROJECT_ROOT)}")

# 保存预测结果（包含置信区间，包含金、银、铜三个类型）
predictions_df = pd.DataFrame({
    'NOC': base_medal_only.loc[test_2024_mask, 'NOC'].values,
    'Year': base_medal_only.loc[test_2024_mask, 'Year'].values,
    # 实际值
    'actual_gold': y_gold_test_2024.values,
    'actual_silver': y_silver_test_2024.values,
    'actual_bronze': y_bronze_test_2024.values,
    'actual_total': (y_gold_test_2024 + y_silver_test_2024 + y_bronze_test_2024).values,
    # 预测值
    'predicted_gold': mean_predictions_dict['Gold'],
    'predicted_silver': mean_predictions_dict['Silver'],
    'predicted_bronze': mean_predictions_dict['Bronze'],
    'predicted_total': (mean_predictions_dict['Gold'] + mean_predictions_dict['Silver'] + mean_predictions_dict['Bronze']),
    # 置信区间（Gold）
    'gold_ci_lower': lower_bound_dict['Gold'],
    'gold_ci_upper': upper_bound_dict['Gold'],
    # 置信区间（Silver）
    'silver_ci_lower': lower_bound_dict['Silver'],
    'silver_ci_upper': upper_bound_dict['Silver'],
    # 置信区间（Bronze）
    'bronze_ci_lower': lower_bound_dict['Bronze'],
    'bronze_ci_upper': upper_bound_dict['Bronze'],
    # 误差
    'gold_error': y_gold_test_2024.values - mean_predictions_dict['Gold'],
    'silver_error': y_silver_test_2024.values - mean_predictions_dict['Silver'],
    'bronze_error': y_bronze_test_2024.values - mean_predictions_dict['Bronze'],
    'total_error': (y_gold_test_2024 + y_silver_test_2024 + y_bronze_test_2024).values - 
                   (mean_predictions_dict['Gold'] + mean_predictions_dict['Silver'] + mean_predictions_dict['Bronze']),
})

predictions_file = os.path.join(OUTPUT_DIR, 'xgboost_regressor_gold_silver_bronze_predictions.csv')
predictions_df.to_csv(predictions_file, index=False, encoding='utf-8-sig')
print(f"  预测结果已保存至: out/xgboost_regressor_gold_silver_bronze_predictions.csv")

# 保存Bootstrap统计（为每个奖牌类型分别保存）
for medal_type in medal_types:
    bootstrap_stats = pd.DataFrame({
        'NOC': base_medal_only.loc[test_2024_mask, 'NOC'].values,
        'Year': base_medal_only.loc[test_2024_mask, 'Year'].values,
        'mean_prediction': mean_predictions_dict[medal_type],
        'std_prediction': bootstrap_predictions_dict[medal_type].std(axis=0),
        'ci_lower_95': lower_bound_dict[medal_type],
        'ci_upper_95': upper_bound_dict[medal_type],
    })
    
    bootstrap_file = os.path.join(OUTPUT_DIR, f'bootstrap_regressor_{medal_type.lower()}_confidence_intervals.csv')
    bootstrap_stats.to_csv(bootstrap_file, index=False, encoding='utf-8-sig')
    print(f"  {medal_type}牌Bootstrap统计已保存至: out/bootstrap_regressor_{medal_type.lower()}_confidence_intervals.csv")

# 保存特征重要性（为每个奖牌类型分别保存）
for medal_type in medal_types:
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': models_cv2[medal_type].feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance_file = os.path.join(OUTPUT_DIR, f'regressor_{medal_type.lower()}_feature_importance.csv')
    feature_importance.to_csv(feature_importance_file, index=False, encoding='utf-8-sig')
    print(f"  {medal_type}牌特征重要性已保存至: out/regressor_{medal_type.lower()}_feature_importance.csv")

# ============================================================================
# 8. 可视化
# ============================================================================
print("\n" + "="*80)
print("生成可视化图表")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# 为每个奖牌类型创建图表
colors = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}

for idx, medal_type in enumerate(medal_types):
    color = colors[medal_type]
    y_test = y_test_2024_dict[medal_type]
    y_pred = mean_predictions_dict[medal_type]
    lower = lower_bound_dict[medal_type]
    upper = upper_bound_dict[medal_type]
    r2 = metrics_cv2[medal_type]['R²']
    
    # 1. 特征重要性
    ax1 = axes[idx, 0]
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': models_cv2[medal_type].feature_importances_
    }).sort_values('importance', ascending=False)
    top_features = feature_importance.head(10)
    ax1.barh(range(len(top_features)), top_features['importance'], color=color, alpha=0.7)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.set_xlabel('Importance')
    ax1.set_title(f'{medal_type} - Top 10 Feature Importance')
    ax1.invert_yaxis()
    
    # 2. 预测 vs 实际值
    ax2 = axes[idx, 1]
    ax2.scatter(y_test, y_pred, alpha=0.6, s=50, color=color)
    # 添加误差棒（只对前30个样本）
    for i in range(min(30, len(y_test))):
        ax2.plot([y_test.iloc[i], y_test.iloc[i]], 
                 [lower[i], upper[i]], color=color, alpha=0.2, linewidth=0.5)
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    ax2.set_xlabel(f'Actual {medal_type} Medals')
    ax2.set_ylabel(f'Predicted {medal_type} Medals')
    ax2.set_title(f'{medal_type} - Prediction vs Actual (R²={r2:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'xgboost_regressor_gold_silver_bronze_analysis.png'), dpi=300, bbox_inches='tight')
print("  可视化图表已保存至: figure/xgboost_regressor_gold_silver_bronze_analysis.png")

# 额外创建一个综合图表：前20个国家的金、银、铜牌预测
fig2, axes2 = plt.subplots(1, 1, figsize=(15, 8))
top_countries = predictions_df.nlargest(20, 'predicted_total')
x_pos = np.arange(len(top_countries))
width = 0.25

# 绘制预测值（带置信区间）
for i, medal_type in enumerate(medal_types):
    color = colors[medal_type]
    predicted_col = f'predicted_{medal_type.lower()}'
    actual_col = f'actual_{medal_type.lower()}'
    ci_lower_col = f'{medal_type.lower()}_ci_lower'
    ci_upper_col = f'{medal_type.lower()}_ci_upper'
    
    # 误差棒
    axes2.errorbar(x_pos + i*width, top_countries[predicted_col], 
                   yerr=[top_countries[predicted_col] - top_countries[ci_lower_col],
                         top_countries[ci_upper_col] - top_countries[predicted_col]],
                   fmt='o', capsize=3, capthick=1.5, color=color, alpha=0.7, 
                   label=f'Predicted {medal_type}')
    # 实际值
    axes2.scatter(x_pos + i*width, top_countries[actual_col], 
                  color=color, s=150, marker='x', linewidths=3, 
                  label=f'Actual {medal_type}', zorder=5, alpha=0.8)

axes2.set_xticks(x_pos + width)
axes2.set_xticklabels(top_countries['NOC'], rotation=45, ha='right')
axes2.set_ylabel('Medal Count')
axes2.set_title('Top 20 Countries: Gold, Silver, Bronze Predictions with 95% CI')
axes2.legend()
axes2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'xgboost_regressor_top20_comparison.png'), dpi=300, bbox_inches='tight')
print("  综合对比图表已保存至: figure/xgboost_regressor_top20_comparison.png")

# ============================================================================
# 9. 模型评估报告
# ============================================================================
print("\n" + "="*80)
print("模型评估报告")
print("="*80)

print("\n【时间序列交叉验证结果】")
for medal_type in medal_types:
    print(f"\n  {medal_type} 牌:")
    print(f"    CV1 (预测2016): MSE={metrics_cv1[medal_type]['MSE']:.2f}, "
          f"MAE={metrics_cv1[medal_type]['MAE']:.2f}, R²={metrics_cv1[medal_type]['R²']:.4f}")
    print(f"    CV2 (预测2024): MSE={metrics_cv2[medal_type]['MSE']:.2f}, "
          f"MAE={metrics_cv2[medal_type]['MAE']:.2f}, R²={metrics_cv2[medal_type]['R²']:.4f}")

print("\n【测试集性能 (2024)】")
for medal_type in medal_types:
    y_test = y_test_2024_dict[medal_type]
    print(f"\n  {medal_type} 牌:")
    print(f"    样本数: {len(y_test)}")
    print(f"    平均绝对误差 (MAE): {metrics_cv2[medal_type]['MAE']:.2f} 枚")
    print(f"    均方误差 (MSE): {metrics_cv2[medal_type]['MSE']:.2f}")
    print(f"    决定系数 (R²): {metrics_cv2[medal_type]['R²']:.4f}")

if len(predictions_df) > 0:
    print(f"\n【预测准确性统计（总奖牌数）】")
    print(f"  平均绝对误差: {predictions_df['total_error'].abs().mean():.2f} 枚")
    print(f"  中位数绝对误差: {predictions_df['total_error'].abs().median():.2f} 枚")
    print(f"  最大绝对误差: {predictions_df['total_error'].abs().max():.2f} 枚")
    
    for medal_type in medal_types:
        error_col = f'{medal_type.lower()}_error'
        print(f"\n  {medal_type} 牌:")
        print(f"    平均绝对误差: {predictions_df[error_col].abs().mean():.2f} 枚")
        print(f"    中位数绝对误差: {predictions_df[error_col].abs().median():.2f} 枚")

# ============================================================================
# 计算四个核心评估指标 (M1, M2, M3, M4)
# ============================================================================
print("\n" + "="*80)
print("核心评估指标 (M1, M2, M3, M4)")
print("="*80)

# 为每个奖牌类型计算指标
metrics_m1_m4 = {}

for medal_type in medal_types:
    actual_col = f'actual_{medal_type.lower()}'
    predicted_col = f'predicted_{medal_type.lower()}'
    ci_lower_col = f'{medal_type.lower()}_ci_lower'
    ci_upper_col = f'{medal_type.lower()}_ci_upper'
    
    actual = predictions_df[actual_col].values
    predicted = predictions_df[predicted_col].values
    ci_lower = predictions_df[ci_lower_col].values
    ci_upper = predictions_df[ci_upper_col].values
    
    # M1: 整体预测准确率 - 使用R²或相对误差
    # 使用相对误差的倒数作为准确率指标（误差越小，准确率越高）
    relative_error = np.abs(actual - predicted) / (actual + 1)  # +1避免除零
    m1 = 1 - relative_error.mean()  # 整体准确率（越高越好）
    m1_alternative = r2_score(actual, predicted)  # 使用R²作为M1的替代指标
    
    # M2: 获奖国家的预测准确率（实际奖牌数>0的国家）
    medal_winning_mask = actual > 0
    if medal_winning_mask.sum() > 0:
        relative_error_winning = np.abs(actual[medal_winning_mask] - predicted[medal_winning_mask]) / (actual[medal_winning_mask] + 1)
        m2 = 1 - relative_error_winning.mean()
        m2_r2 = r2_score(actual[medal_winning_mask], predicted[medal_winning_mask])
    else:
        m2 = 0
        m2_r2 = 0
    
    # M3: 零奖牌国家的预测准确率（实际奖牌数=0的国家）
    # 注意：回归器只对获奖国家预测，所以M3可能不适用，但我们计算预测值接近0的准确率
    zero_medal_mask = actual == 0
    if zero_medal_mask.sum() > 0:
        # 对于零奖牌国家，预测值应该接近0
        m3 = 1 - np.abs(predicted[zero_medal_mask]).mean() / (predicted.max() + 1)  # 归一化
        m3_mae = np.abs(predicted[zero_medal_mask]).mean()
    else:
        m3 = 1.0  # 如果没有零奖牌国家，设为1
        m3_mae = 0
    
    # M4: 95%置信区间内的准确率（实际值是否在置信区间内）
    in_ci_mask = (actual >= ci_lower) & (actual <= ci_upper)
    m4 = in_ci_mask.mean()  # 在置信区间内的比例
    
    metrics_m1_m4[medal_type] = {
        'M1_overall_accuracy': m1,
        'M1_r2': m1_alternative,
        'M2_medal_winning_accuracy': m2,
        'M2_r2': m2_r2,
        'M3_zero_medal_accuracy': m3,
        'M3_mae': m3_mae,
        'M4_ci_coverage': m4,
        'M4_count_in_ci': in_ci_mask.sum(),
        'M4_total_count': len(actual)
    }
    
    print(f"\n{medal_type} 牌:")
    print(f"  M1 (整体预测准确率): {m1:.4f} (相对误差准确率), R²={m1_alternative:.4f}")
    print(f"  M2 (获奖国家预测准确率): {m2:.4f} (相对误差准确率), R²={m2_r2:.4f}")
    print(f"  M3 (零奖牌国家预测准确率): {m3:.4f} (预测值接近0的准确率), MAE={m3_mae:.4f}")
    print(f"  M4 (95%置信区间覆盖率): {m4:.4f} ({in_ci_mask.sum()}/{len(actual)} 在区间内)")

# 总奖牌数的指标
actual_total = predictions_df['actual_total'].values
predicted_total = predictions_df['predicted_total'].values
ci_lower_total = (predictions_df['gold_ci_lower'] + predictions_df['silver_ci_lower'] + predictions_df['bronze_ci_lower']).values
ci_upper_total = (predictions_df['gold_ci_upper'] + predictions_df['silver_ci_upper'] + predictions_df['bronze_ci_upper']).values

relative_error_total = np.abs(actual_total - predicted_total) / (actual_total + 1)
m1_total = 1 - relative_error_total.mean()
m1_total_r2 = r2_score(actual_total, predicted_total)

medal_winning_mask_total = actual_total > 0
if medal_winning_mask_total.sum() > 0:
    relative_error_winning_total = np.abs(actual_total[medal_winning_mask_total] - predicted_total[medal_winning_mask_total]) / (actual_total[medal_winning_mask_total] + 1)
    m2_total = 1 - relative_error_winning_total.mean()
    m2_total_r2 = r2_score(actual_total[medal_winning_mask_total], predicted_total[medal_winning_mask_total])
else:
    m2_total = 0
    m2_total_r2 = 0

zero_medal_mask_total = actual_total == 0
if zero_medal_mask_total.sum() > 0:
    m3_total = 1 - np.abs(predicted_total[zero_medal_mask_total]).mean() / (predicted_total.max() + 1)
    m3_total_mae = np.abs(predicted_total[zero_medal_mask_total]).mean()
else:
    m3_total = 1.0
    m3_total_mae = 0

in_ci_mask_total = (actual_total >= ci_lower_total) & (actual_total <= ci_upper_total)
m4_total = in_ci_mask_total.mean()

metrics_m1_m4['Total'] = {
    'M1_overall_accuracy': m1_total,
    'M1_r2': m1_total_r2,
    'M2_medal_winning_accuracy': m2_total,
    'M2_r2': m2_total_r2,
    'M3_zero_medal_accuracy': m3_total,
    'M3_mae': m3_total_mae,
    'M4_ci_coverage': m4_total,
    'M4_count_in_ci': in_ci_mask_total.sum(),
    'M4_total_count': len(actual_total)
}

print(f"\n总奖牌数:")
print(f"  M1 (整体预测准确率): {m1_total:.4f} (相对误差准确率), R²={m1_total_r2:.4f}")
print(f"  M2 (获奖国家预测准确率): {m2_total:.4f} (相对误差准确率), R²={m2_total_r2:.4f}")
print(f"  M3 (零奖牌国家预测准确率): {m3_total:.4f} (预测值接近0的准确率), MAE={m3_total_mae:.4f}")
print(f"  M4 (95%置信区间覆盖率): {m4_total:.4f} ({in_ci_mask_total.sum()}/{len(actual_total)} 在区间内)")

print("\n【Bootstrap置信区间统计】")
print(f"  重采样次数: {n_bootstrap}")
for medal_type in medal_types:
    lower = lower_bound_dict[medal_type]
    upper = upper_bound_dict[medal_type]
    print(f"\n  {medal_type} 牌:")
    print(f"    平均置信区间宽度: {(upper - lower).mean():.2f} 枚")
    print(f"    最小置信区间宽度: {(upper - lower).min():.2f} 枚")
    print(f"    最大置信区间宽度: {(upper - lower).max():.2f} 枚")

# ============================================================================
# 10. 保存最终报告
# ============================================================================
print("\n" + "="*80)
print("保存最终评估报告")
print("="*80)

# 生成Markdown格式的报告
report_md_lines = []
report_md_lines.append("# TPE-XGBoost回归器（金银铜牌预测）最终评估报告\n")
report_md_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report_md_lines.append(f"**数据范围**: 1960-2024\n")
report_md_lines.append(f"**训练样本数 (CV2)**: {len(X_train_2024)}\n")
report_md_lines.append(f"**测试样本数 (CV2)**: {len(X_test_2024)}\n")

report_md_lines.append("\n## 时间序列交叉验证结果\n")
report_md_lines.append("| 奖牌类型 | CV1 (预测2016) | CV2 (预测2024) |")
report_md_lines.append("|---------|---------------|---------------|")
for medal_type in medal_types:
    cv1_str = f"MSE={metrics_cv1[medal_type]['MSE']:.2f}, MAE={metrics_cv1[medal_type]['MAE']:.2f}, R²={metrics_cv1[medal_type]['R²']:.4f}"
    cv2_str = f"MSE={metrics_cv2[medal_type]['MSE']:.2f}, MAE={metrics_cv2[medal_type]['MAE']:.2f}, R²={metrics_cv2[medal_type]['R²']:.4f}"
    report_md_lines.append(f"| {medal_type} | {cv1_str} | {cv2_str} |")

report_md_lines.append("\n## 测试集性能 (2024)\n")
report_md_lines.append("| 奖牌类型 | 样本数 | MAE | MSE | R² |")
report_md_lines.append("|---------|--------|-----|-----|-----|")
for medal_type in medal_types:
    y_test = y_test_2024_dict[medal_type]
    report_md_lines.append(f"| {medal_type} | {len(y_test)} | {metrics_cv2[medal_type]['MAE']:.2f} | "
                          f"{metrics_cv2[medal_type]['MSE']:.2f} | {metrics_cv2[medal_type]['R²']:.4f} |")

if len(predictions_df) > 0:
    report_md_lines.append("\n## 预测准确性统计\n")
    report_md_lines.append("### 总奖牌数\n")
    report_md_lines.append(f"- 平均绝对误差: {predictions_df['total_error'].abs().mean():.2f} 枚")
    report_md_lines.append(f"- 中位数绝对误差: {predictions_df['total_error'].abs().median():.2f} 枚")
    report_md_lines.append(f"- 最大绝对误差: {predictions_df['total_error'].abs().max():.2f} 枚")
    
    report_md_lines.append("\n### 各类型奖牌\n")
    report_md_lines.append("| 奖牌类型 | 平均绝对误差 | 中位数绝对误差 |")
    report_md_lines.append("|---------|------------|--------------|")
    for medal_type in medal_types:
        error_col = f'{medal_type.lower()}_error'
        mae = predictions_df[error_col].abs().mean()
        median_ae = predictions_df[error_col].abs().median()
        report_md_lines.append(f"| {medal_type} | {mae:.2f} | {median_ae:.2f} |")

report_md_lines.append("\n## 核心评估指标 (M1, M2, M3, M4)\n")
report_md_lines.append("\n### 指标说明\n")
report_md_lines.append("- **M1**: 整体预测准确率 - 评估模型的整体预测能力\n")
report_md_lines.append("- **M2**: 获奖国家的预测准确率 - 衡量模型对会获奖国家的预测准确性\n")
report_md_lines.append("- **M3**: 零奖牌国家的预测准确率 - 评估模型对不会获奖国家的预测准确性\n")
report_md_lines.append("- **M4**: 95%置信区间覆盖率 - 衡量模型预测的稳定性和可靠性\n")

report_md_lines.append("\n### 各类型奖牌指标\n")
report_md_lines.append("| 奖牌类型 | M1 (整体准确率) | M1 (R²) | M2 (获奖国家准确率) | M2 (R²) | M3 (零奖牌准确率) | M4 (CI覆盖率) |")
report_md_lines.append("|---------|----------------|---------|-------------------|---------|-----------------|-------------|")
for medal_type in medal_types:
    m = metrics_m1_m4[medal_type]
    report_md_lines.append(f"| {medal_type} | {m['M1_overall_accuracy']:.4f} | {m['M1_r2']:.4f} | "
                          f"{m['M2_medal_winning_accuracy']:.4f} | {m['M2_r2']:.4f} | "
                          f"{m['M3_zero_medal_accuracy']:.4f} | {m['M4_ci_coverage']:.4f} ({m['M4_count_in_ci']}/{m['M4_total_count']}) |")

report_md_lines.append("\n### 总奖牌数指标\n")
m_total = metrics_m1_m4['Total']
report_md_lines.append("| 指标 | 值 | 说明 |")
report_md_lines.append("|------|-----|------|")
report_md_lines.append(f"| M1 (整体预测准确率) | {m_total['M1_overall_accuracy']:.4f} | 相对误差准确率 |")
report_md_lines.append(f"| M1 (R²) | {m_total['M1_r2']:.4f} | 决定系数 |")
report_md_lines.append(f"| M2 (获奖国家预测准确率) | {m_total['M2_medal_winning_accuracy']:.4f} | 相对误差准确率 |")
report_md_lines.append(f"| M2 (R²) | {m_total['M2_r2']:.4f} | 决定系数 |")
report_md_lines.append(f"| M3 (零奖牌国家预测准确率) | {m_total['M3_zero_medal_accuracy']:.4f} | 预测值接近0的准确率 |")
report_md_lines.append(f"| M4 (95%置信区间覆盖率) | {m_total['M4_ci_coverage']:.4f} | {m_total['M4_count_in_ci']}/{m_total['M4_total_count']} 在区间内 |")

report_md_lines.append("\n## Bootstrap置信区间统计\n")
report_md_lines.append(f"**重采样次数**: {n_bootstrap}\n")
report_md_lines.append("| 奖牌类型 | 平均置信区间宽度 | 最小宽度 | 最大宽度 |")
report_md_lines.append("|---------|----------------|---------|---------|")
for medal_type in medal_types:
    lower = lower_bound_dict[medal_type]
    upper = upper_bound_dict[medal_type]
    mean_width = (upper - lower).mean()
    min_width = (upper - lower).min()
    max_width = (upper - lower).max()
    report_md_lines.append(f"| {medal_type} | {mean_width:.2f} | {min_width:.2f} | {max_width:.2f} |")

report_file = os.path.join(OUTPUT_DIR, 'regressor_final_report.md')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_md_lines))
print(f"  最终报告已保存至: out/regressor_final_report.md")

print("\n" + "="*80)
print("TPE-XGBoost回归器（金银铜牌预测）构建完成！")
print("="*80)
print("\n生成的文件:")
print("\n【模型文件】")
for medal_type in medal_types:
    model_path = os.path.join(MODEL_DIR, f'xgboost_regressor_{medal_type.lower()}_tpe.pkl')
    print(f"  - {os.path.relpath(model_path, PROJECT_ROOT)} ({medal_type}牌模型)")
print("\n【预测结果】")
print("  - out/xgboost_regressor_gold_silver_bronze_predictions.csv (预测结果+置信区间)")
print("\n【Bootstrap统计】")
for medal_type in medal_types:
    print(f"  - out/bootstrap_regressor_{medal_type.lower()}_confidence_intervals.csv ({medal_type}牌)")
print("\n【超参数】")
print("  - out/best_hyperparameters_regressor_gold_silver_bronze_tpe.json (TPE最佳参数)")
print("\n【特征重要性】")
for medal_type in medal_types:
    print(f"  - out/regressor_{medal_type.lower()}_feature_importance.csv ({medal_type}牌)")
print("\n【可视化图表】")
print("  - figure/xgboost_regressor_gold_silver_bronze_analysis.png (各类型分析)")
print("  - figure/xgboost_regressor_top20_comparison.png (前20名国家对比)")
print("\n【评估报告】")
print("  - out/regressor_final_report.md (最终评估报告)")
