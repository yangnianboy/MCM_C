"""
TPE-XGBoost回归器
- 对分类器预测为1的样本（获得奖牌的国家），预测具体的奖牌数量
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
print("TPE-XGBoost回归器 - 预测奖牌数量")
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
y = base_medal_only['Total']  # 预测总奖牌数

print(f"  总样本数: {len(X)} (只包含获得奖牌的国家)")
print(f"  奖牌数范围: [{y.min()}, {y.max()}]")
print(f"  平均奖牌数: {y.mean():.2f}")
print(f"  中位数奖牌数: {y.median():.2f}")

# 时间序列交叉验证的数据分割
# CV1: 用1960-2012训练，预测2016
train_2016_mask = (base_medal_only['Year'] >= 1960) & (base_medal_only['Year'] <= 2012)
test_2016_mask = base_medal_only['Year'] == 2016

# CV2: 用1960-2016训练，预测2024
train_2024_mask = (base_medal_only['Year'] >= 1960) & (base_medal_only['Year'] <= 2016)
test_2024_mask = base_medal_only['Year'] == 2024

X_train_2016 = X[train_2016_mask]
y_train_2016 = y[train_2016_mask]
X_test_2016 = X[test_2016_mask]
y_test_2016 = y[test_2016_mask]

X_train_2024 = X[train_2024_mask]
y_train_2024 = y[train_2024_mask]
X_test_2024 = X[test_2024_mask]
y_test_2024 = y[test_2024_mask]

print(f"\n  时间序列交叉验证设置:")
print(f"    CV1: 训练(1960-2012, {len(X_train_2016)}样本) -> 测试(2016, {len(X_test_2016)}样本)")
print(f"    CV2: 训练(1960-2016, {len(X_train_2024)}样本) -> 测试(2024, {len(X_test_2024)}样本)")

# ============================================================================
# 4. TPE超参数搜索
# ============================================================================
print("\n[4/7] TPE超参数搜索...")

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

def objective(params):
    """TPE优化的目标函数 - 使用MSE作为损失"""
    # 转换参数类型
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    params['min_child_weight'] = int(params['min_child_weight'])
    params['objective'] = 'reg:squarederror'  # 回归任务
    params['eval_metric'] = 'rmse'
    params['random_state'] = 42
    params['n_jobs'] = -1
    
    # 时间序列交叉验证：使用2016训练集训练，2016测试集验证
    model = xgb.XGBRegressor(**params)
    
    try:
        model.fit(X_train_2016, y_train_2016, verbose=False)
        y_pred = model.predict(X_test_2016)
        
        # 使用MSE作为损失
        mse = mean_squared_error(y_test_2016, y_pred)
        
        return {'loss': mse, 'status': STATUS_OK}
    except Exception as e:
        return {'loss': 1e6, 'status': STATUS_OK}

# 执行TPE搜索
print("  开始TPE超参数搜索（50次迭代）...")
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials,
    verbose=1
)

# 转换最佳参数
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

print("\n  最佳超参数 (TPE搜索):")
for param, value in sorted(best_params.items()):
    if param not in ['objective', 'eval_metric', 'random_state', 'n_jobs']:
        print(f"    {param}: {value:.4f}" if isinstance(value, float) else f"    {param}: {value}")

# 保存最佳参数
best_params_file = os.path.join(OUTPUT_DIR, 'best_hyperparameters_regressor_tpe.json')
with open(best_params_file, 'w', encoding='utf-8') as f:
    json.dump({k: float(v) if isinstance(v, (int, float)) else v 
               for k, v in best_params.items() if k not in ['objective', 'eval_metric', 'random_state', 'n_jobs']}, 
              f, indent=2, ensure_ascii=False)
print(f"\n  最佳参数已保存至: out/best_hyperparameters_regressor_tpe.json")

# ============================================================================
# 5. 时间序列交叉验证训练
# ============================================================================
print("\n[5/7] 时间序列交叉验证训练...")

# CV1: 用1960-2012训练，预测2016
print("\n  CV1: 训练(1960-2012) -> 预测(2016)")
model_cv1 = xgb.XGBRegressor(**best_params)
model_cv1.fit(X_train_2016, y_train_2016, verbose=False)
y_pred_cv1 = model_cv1.predict(X_test_2016)

mse_cv1 = mean_squared_error(y_test_2016, y_pred_cv1)
mae_cv1 = mean_absolute_error(y_test_2016, y_pred_cv1)
r2_cv1 = r2_score(y_test_2016, y_pred_cv1) if len(y_test_2016) > 1 else 0

print(f"    MSE: {mse_cv1:.2f}")
print(f"    MAE: {mae_cv1:.2f}")
print(f"    R²: {r2_cv1:.4f}")

# CV2: 用1960-2016训练，预测2024
print("\n  CV2: 训练(1960-2016) -> 预测(2024)")
model_cv2 = xgb.XGBRegressor(**best_params)
model_cv2.fit(X_train_2024, y_train_2024, verbose=False)
y_pred_cv2 = model_cv2.predict(X_test_2024)

mse_cv2 = mean_squared_error(y_test_2024, y_pred_cv2)
mae_cv2 = mean_absolute_error(y_test_2024, y_pred_cv2)
r2_cv2 = r2_score(y_test_2024, y_pred_cv2) if len(y_test_2024) > 1 else 0

print(f"    MSE: {mse_cv2:.2f}")
print(f"    MAE: {mae_cv2:.2f}")
print(f"    R²: {r2_cv2:.4f}")

# ============================================================================
# 6. Bootstrap置信区间
# ============================================================================
print("\n[6/7] Bootstrap置信区间计算...")

n_bootstrap = 1000
bootstrap_predictions = []

print(f"  执行 {n_bootstrap} 次Bootstrap重采样...")

for i in range(n_bootstrap):
    if (i + 1) % 100 == 0:
        print(f"    进度: {i+1}/{n_bootstrap}")
    
    # 重采样训练数据
    indices = np.random.choice(len(X_train_2024), size=len(X_train_2024), replace=True)
    X_boot = X_train_2024.iloc[indices]
    y_boot = y_train_2024.iloc[indices]
    
    # 训练模型
    model_boot = xgb.XGBRegressor(**best_params)
    model_boot.fit(X_boot, y_boot, verbose=False)
    
    # 预测测试集
    y_pred_boot = model_boot.predict(X_test_2024)
    bootstrap_predictions.append(y_pred_boot)

bootstrap_predictions = np.array(bootstrap_predictions)

# 计算置信区间
mean_predictions = bootstrap_predictions.mean(axis=0)
lower_bound = np.percentile(bootstrap_predictions, 2.5, axis=0)  # 95%置信区间下界
upper_bound = np.percentile(bootstrap_predictions, 97.5, axis=0)  # 95%置信区间上界

print(f"  Bootstrap完成！")
print(f"  预测均值范围: [{mean_predictions.min():.2f}, {mean_predictions.max():.2f}]")
print(f"  平均置信区间宽度: {(upper_bound - lower_bound).mean():.2f}")

# ============================================================================
# 7. 保存结果
# ============================================================================
print("\n" + "="*80)
print("保存结果")
print("="*80)

# 保存最终模型（使用CV2的模型）
final_model = model_cv2
model_path = os.path.join(MODEL_DIR, 'xgboost_regressor_tpe.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"  模型已保存至: {os.path.relpath(model_path, PROJECT_ROOT)}")

# 保存预测结果（包含置信区间）
predictions_df = pd.DataFrame({
    'NOC': base_medal_only.loc[test_2024_mask, 'NOC'].values,
    'Year': base_medal_only.loc[test_2024_mask, 'Year'].values,
    'actual_medals': y_test_2024.values,
    'predicted_medals': mean_predictions,
    'ci_lower': lower_bound,
    'ci_upper': upper_bound,
    'ci_width': upper_bound - lower_bound,
    'error': y_test_2024.values - mean_predictions,
    'abs_error': np.abs(y_test_2024.values - mean_predictions)
})

predictions_file = os.path.join(OUTPUT_DIR, 'xgboost_regressor_predictions.csv')
predictions_df.to_csv(predictions_file, index=False, encoding='utf-8-sig')
print(f"  预测结果已保存至: out/xgboost_regressor_predictions.csv")

# 保存Bootstrap统计
bootstrap_stats = pd.DataFrame({
    'NOC': base_medal_only.loc[test_2024_mask, 'NOC'].values,
    'Year': base_medal_only.loc[test_2024_mask, 'Year'].values,
    'mean_prediction': mean_predictions,
    'std_prediction': bootstrap_predictions.std(axis=0),
    'ci_lower_95': lower_bound,
    'ci_upper_95': upper_bound,
})

bootstrap_file = os.path.join(OUTPUT_DIR, 'bootstrap_regressor_confidence_intervals.csv')
bootstrap_stats.to_csv(bootstrap_file, index=False, encoding='utf-8-sig')
print(f"  Bootstrap统计已保存至: out/bootstrap_regressor_confidence_intervals.csv")

# 保存特征重要性
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance_file = os.path.join(OUTPUT_DIR, 'regressor_feature_importance.csv')
feature_importance.to_csv(feature_importance_file, index=False, encoding='utf-8-sig')
print(f"  特征重要性已保存至: out/regressor_feature_importance.csv")

# ============================================================================
# 8. 可视化
# ============================================================================
print("\n" + "="*80)
print("生成可视化图表")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 特征重要性
ax1 = axes[0, 0]
top_features = feature_importance.head(15)
ax1.barh(range(len(top_features)), top_features['importance'])
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'])
ax1.set_xlabel('Importance')
ax1.set_title('Top 15 Feature Importance')
ax1.invert_yaxis()

# 2. 预测 vs 实际值
ax2 = axes[0, 1]
ax2.scatter(y_test_2024, mean_predictions, alpha=0.6, s=50)
# 添加误差棒
for i in range(len(y_test_2024)):
    ax2.plot([y_test_2024.iloc[i], y_test_2024.iloc[i]], 
             [lower_bound[i], upper_bound[i]], 'g-', alpha=0.3, linewidth=1)
ax2.plot([y_test_2024.min(), y_test_2024.max()], 
         [y_test_2024.min(), y_test_2024.max()], 'r--', label='Perfect Prediction')
ax2.set_xlabel('Actual Medals')
ax2.set_ylabel('Predicted Medals')
ax2.set_title(f'Prediction vs Actual (R²={r2_cv2:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 置信区间可视化（前20个国家）
ax3 = axes[1, 0]
top_countries = predictions_df.nlargest(20, 'predicted_medals')
x_pos = np.arange(len(top_countries))
ax3.errorbar(x_pos, top_countries['predicted_medals'], 
             yerr=[top_countries['predicted_medals'] - top_countries['ci_lower'],
                   top_countries['ci_upper'] - top_countries['predicted_medals']],
             fmt='o', capsize=5, capthick=2, label='Predicted with 95% CI')
ax3.scatter(x_pos, top_countries['actual_medals'], color='red', s=100, 
            marker='x', linewidths=2, label='Actual', zorder=5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(top_countries['NOC'], rotation=45, ha='right')
ax3.set_ylabel('Medal Count')
ax3.set_title('Top 20 Countries: Prediction with 95% CI')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Bootstrap分布（选择一个示例国家）
ax4 = axes[1, 1]
if len(predictions_df) > 0:
    example_idx = 0
    example_noc = predictions_df.iloc[example_idx]['NOC']
    example_dist = bootstrap_predictions[:, example_idx]
    ax4.hist(example_dist, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(mean_predictions[example_idx], color='r', linestyle='--', 
                linewidth=2, label='Mean')
    ax4.axvline(lower_bound[example_idx], color='g', linestyle='--', 
                linewidth=2, label='95% CI')
    ax4.axvline(upper_bound[example_idx], color='g', linestyle='--', linewidth=2)
    if len(y_test_2024) > example_idx:
        ax4.axvline(y_test_2024.iloc[example_idx], color='orange', 
                    linestyle='-', linewidth=2, label='Actual')
    ax4.set_xlabel('Predicted Medal Count')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Bootstrap Distribution: {example_noc}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'xgboost_regressor_analysis.png'), dpi=300, bbox_inches='tight')
print("  可视化图表已保存至: figure/xgboost_regressor_analysis.png")

# ============================================================================
# 9. 模型评估报告
# ============================================================================
print("\n" + "="*80)
print("模型评估报告")
print("="*80)

print("\n【时间序列交叉验证结果】")
print(f"  CV1 (预测2016): MSE={mse_cv1:.2f}, MAE={mae_cv1:.2f}, R²={r2_cv1:.4f}")
print(f"  CV2 (预测2024): MSE={mse_cv2:.2f}, MAE={mae_cv2:.2f}, R²={r2_cv2:.4f}")

print("\n【测试集性能 (2024)】")
print(f"  样本数: {len(y_test_2024)}")
print(f"  平均绝对误差 (MAE): {mae_cv2:.2f} 枚奖牌")
print(f"  均方误差 (MSE): {mse_cv2:.2f}")
print(f"  决定系数 (R²): {r2_cv2:.4f}")

if len(predictions_df) > 0:
    print(f"\n【预测准确性统计】")
    print(f"  平均绝对误差: {predictions_df['abs_error'].mean():.2f} 枚")
    print(f"  中位数绝对误差: {predictions_df['abs_error'].median():.2f} 枚")
    print(f"  最大绝对误差: {predictions_df['abs_error'].max():.2f} 枚")

print("\n【Bootstrap置信区间统计】")
print(f"  重采样次数: {n_bootstrap}")
print(f"  平均置信区间宽度: {(upper_bound - lower_bound).mean():.2f} 枚")
print(f"  最小置信区间宽度: {(upper_bound - lower_bound).min():.2f} 枚")
print(f"  最大置信区间宽度: {(upper_bound - lower_bound).max():.2f} 枚")

print("\n" + "="*80)
print("TPE-XGBoost回归器构建完成！")
print("="*80)
print("\n生成的文件:")
print(f"  - {os.path.relpath(model_path, PROJECT_ROOT)} (回归模型)")
print("  - out/xgboost_regressor_predictions.csv (预测结果+置信区间)")
print("  - out/bootstrap_regressor_confidence_intervals.csv (Bootstrap统计)")
print("  - out/best_hyperparameters_regressor_tpe.json (TPE最佳参数)")
print("  - out/regressor_feature_importance.csv (特征重要性)")
print("  - figure/xgboost_regressor_analysis.png (可视化图表)")
