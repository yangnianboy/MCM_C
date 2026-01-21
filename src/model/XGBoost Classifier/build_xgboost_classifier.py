"""
使用hq_data构建XGBoost分类器
预测某国在某年是否能获得奖牌（二分类：0或1）
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, make_scorer
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 获取项目根目录（脚本在 src/model/XGBoost Classifier/，向上3级）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..', '..')
PROJECT_ROOT = os.path.normpath(PROJECT_ROOT)

# 定义路径
HQ_DATA_DIR = os.path.join(PROJECT_ROOT, 'hq_data')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'out')
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figure')
MODEL_DIR = SCRIPT_DIR  # 模型保存在当前目录

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*80)
print("XGBoost分类器构建 - 预测国家是否能获得奖牌")
print("="*80)
print(f"\n项目根目录: {PROJECT_ROOT}")
print(f"数据目录: {HQ_DATA_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"图表目录: {FIGURE_DIR}")

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n[1/6] 加载数据...")

# 验证文件是否存在
# 使用新生成的包含所有参赛国家的数据（包括未获奖的）
medal_file = os.path.join(HQ_DATA_DIR, 'medal_counts_from_athletes.csv')
athlete_file = os.path.join(HQ_DATA_DIR, 'athletes_cleaned_full.csv')
host_file = os.path.join(DATA_DIR, 'summerOly_hosts.csv')

if not os.path.exists(medal_file):
    raise FileNotFoundError(f"找不到文件: {medal_file}\n请先运行 script/rebuild_medal_counts_from_athletes.py 生成数据")
if not os.path.exists(athlete_file):
    raise FileNotFoundError(f"找不到文件: {athlete_file}")
if not os.path.exists(host_file):
    raise FileNotFoundError(f"找不到文件: {host_file}")

medal_counts = pd.read_csv(medal_file)
athletes = pd.read_csv(athlete_file)
hosts = pd.read_csv(host_file)

print(f"  奖牌数据: {len(medal_counts)} 条 (包含所有参赛国家，未获奖记为0)")
print(f"  运动员数据: {len(athletes)} 条")
print(f"  主办城市数据: {len(hosts)} 条")

# ============================================================================
# 2. 创建基础数据集
# ============================================================================
print("\n[2/6] 创建基础数据集（按国家-年份聚合）...")

# 从medal_counts创建基础框架（新数据已包含所有参赛国家）
base = medal_counts[['NOC', 'Year', 'Gold', 'Silver', 'Bronze', 'Total']].copy()

# 确保Total列正确（如果缺失则计算）
if 'Total' not in base.columns or base['Total'].isna().any():
    base['Total'] = base['Gold'].fillna(0) + base['Silver'].fillna(0) + base['Bronze'].fillna(0)

# 目标变量：是否有奖牌（至少1枚）
base['has_medal'] = (base['Total'] > 0).astype(int)

print(f"  基础记录数: {len(base)} (包含所有参赛国家)")
print(f"  有奖牌: {base['has_medal'].sum()} ({base['has_medal'].mean()*100:.1f}%)")
print(f"  无奖牌: {(base['has_medal']==0).sum()} ({(base['has_medal']==0).mean()*100:.1f}%)")
print(f"  数据完整性: 包含所有参赛国家，未获奖国家Total=0")

# ============================================================================
# 3. 添加参赛规模特征
# ============================================================================
print("\n[3/6] 添加参赛规模特征...")

# 统计每个国家每年的参赛情况
athlete_stats = athletes.groupby(['NOC', 'Year']).agg({
    'Name': 'count',  # 参赛人数
    'Event': 'nunique',  # 参赛项目数
    'Sport': 'nunique',  # 参赛运动大类数
    'is_team_event': 'sum',  # 团体项目数
    'Sex': lambda x: (x == 'F').sum()  # 女性运动员数
}).reset_index()

athlete_stats.columns = ['NOC', 'Year', 'athlete_count', 'event_count', 
                        'sport_count', 'team_events_count', 'female_athletes']

# 合并到基础数据
base = base.merge(athlete_stats, on=['NOC', 'Year'], how='left')

# 填充缺失值（某些国家可能没有参赛记录）
base = base.fillna(0)

# 计算相对规模（相对于该年总参赛人数）
yearly_totals = base.groupby('Year')['athlete_count'].sum()
base['athlete_count_normalized'] = base.apply(
    lambda row: row['athlete_count'] / yearly_totals[row['Year']] if yearly_totals[row['Year']] > 0 else 0,
    axis=1
)

print(f"  参赛规模特征已添加")

# ============================================================================
# 4. 添加历史表现特征
# ============================================================================
print("\n[4/6] 添加历史表现特征...")

# 按国家排序，确保时间顺序
base = base.sort_values(['NOC', 'Year']).reset_index(drop=True)

# 过去N届的奖牌数（lag特征）
for lag in [1, 2, 3]:
    base[f'medal_count_lag{lag}'] = base.groupby('NOC')['Total'].shift(lag)
    base[f'gold_count_lag{lag}'] = base.groupby('NOC')['Gold'].shift(lag)
    base[f'silver_count_lag{lag}'] = base.groupby('NOC')['Silver'].shift(lag)
    base[f'bronze_count_lag{lag}'] = base.groupby('NOC')['Bronze'].shift(lag)

# 过去3届平均奖牌数
base['avg_medals_last_3'] = (
    base['medal_count_lag1'].fillna(0) + 
    base['medal_count_lag2'].fillna(0) + 
    base['medal_count_lag3'].fillna(0)
) / 3

# 历史最高奖牌数（到该年为止）
base['max_medals_historical'] = base.groupby('NOC')['Total'].transform(
    lambda x: x.shift(1).expanding().max()
).fillna(0)

# 历史总奖牌数（到该年为止）
base['total_historical_medals'] = base.groupby('NOC')['Total'].transform(
    lambda x: x.shift(1).expanding().sum()
).fillna(0)

# 历史参赛届数
base['historical_participation_count'] = base.groupby('NOC').cumcount()

# 连续获奖届数（只使用历史数据，shift(1)避免数据泄漏）
def calculate_streak(group):
    """计算连续获奖届数（基于历史数据）"""
    # 使用shift(1)确保只使用历史信息
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

# 距离上次获奖的年数（只使用历史数据）
base_reset = base.reset_index(drop=True)
base_reset['years_since_last_medal'] = 0

for noc in base_reset['NOC'].unique():
    noc_mask = base_reset['NOC'] == noc
    noc_data = base_reset[noc_mask].copy().sort_values('Year')
    
    last_medal_year = None
    for idx, row in noc_data.iterrows():
        # 只使用历史数据（之前年份的has_medal）
        prev_data = noc_data[noc_data['Year'] < row['Year']]
        if len(prev_data) > 0:
            # 检查历史是否有获奖
            prev_medals = prev_data[prev_data['has_medal'] == 1]
            if len(prev_medals) > 0:
                last_medal_year = prev_medals['Year'].iloc[-1]
                base_reset.loc[idx, 'years_since_last_medal'] = row['Year'] - last_medal_year
            else:
                base_reset.loc[idx, 'years_since_last_medal'] = 999  # 历史从未获奖
        else:
            base_reset.loc[idx, 'years_since_last_medal'] = 999  # 首次参赛

base = base_reset.copy()

# 奖牌数趋势（相对于上上届）
base['medal_trend'] = (base['medal_count_lag1'] - base['medal_count_lag2']) / (base['medal_count_lag2'] + 1)
base['medal_trend'] = base['medal_trend'].fillna(0)

# 是否在上升趋势
base['is_improving'] = (base['medal_trend'] > 0).astype(int)

# 填充所有NaN为0
base = base.fillna(0)

print(f"  历史特征已添加")

# ============================================================================
# 5. 添加时间特征
# ============================================================================
print("\n[5/6] 添加时间特征...")

# 年份归一化
base['year_normalized'] = (base['Year'] - 1960) / (2024 - 1960)

# 是否为主办国
# 从hosts数据提取主办国NOC（需要手动映射）
host_mapping = {
    1960: 'ITA',  # Rome
    1964: 'JPN',  # Tokyo
    1968: 'MEX',  # Mexico City
    1972: 'FRG',  # Munich (West Germany)
    1976: 'CAN',  # Montreal
    1988: 'KOR',  # Seoul
    1992: 'ESP',  # Barcelona
    1996: 'USA',  # Atlanta
    2000: 'AUS',  # Sydney
    2004: 'GRE',  # Athens
    2008: 'CHN',  # Beijing
    2012: 'GBR',  # London
    2016: 'BRA',  # Rio de Janeiro
    2024: 'FRA',  # Paris
}

base['is_host'] = base.apply(
    lambda row: 1 if row['NOC'] == host_mapping.get(row['Year'], '') else 0,
    axis=1
)

# 距离首次参赛的年数
first_participation = base.groupby('NOC')['Year'].min()
base['years_since_first_participation'] = base.apply(
    lambda row: row['Year'] - first_participation.get(row['NOC'], row['Year']),
    axis=1
)

print(f"  时间特征已添加")

# ============================================================================
# 6. 添加国家特征
# ============================================================================
print("\n[6/6] 添加国家特征...")

# 国家编码
le_noc = LabelEncoder()
base['NOC_encoded'] = le_noc.fit_transform(base['NOC'])

# 历史获奖率（到该年为止）
base['historical_medal_rate'] = base.groupby('NOC')['has_medal'].transform(
    lambda x: x.shift(1).expanding().mean()
).fillna(0)

# 参赛规模趋势
base['athlete_count_lag1'] = base.groupby('NOC')['athlete_count'].shift(1).fillna(0)
base['athlete_count_trend'] = (base['athlete_count'] - base['athlete_count_lag1']) / (base['athlete_count_lag1'] + 1)

# 交互特征
base['athlete_medal_interaction'] = base['athlete_count'] * base['medal_count_lag1']
base['host_medal_interaction'] = base['is_host'] * base['medal_count_lag1']
base['year_medal_interaction'] = base['year_normalized'] * base['medal_count_lag1']

print(f"  国家特征已添加")

# ============================================================================
# 7. 特征选择
# ============================================================================
print("\n" + "="*80)
print("特征选择")
print("="*80)

feature_cols = [
    # 历史表现特征（最重要）
    'medal_count_lag1', 'medal_count_lag2', 'medal_count_lag3',
    'gold_count_lag1', 'silver_count_lag1', 'bronze_count_lag1',
    'avg_medals_last_3', 'max_medals_historical', 'total_historical_medals',
    'medal_streak', 'years_since_last_medal', 'medal_trend', 'is_improving',
    
    # 参赛规模特征
    'athlete_count', 'event_count', 'sport_count', 'team_events_count',
    'female_athletes', 'athlete_count_normalized', 'athlete_count_trend',
    
    # 时间特征
    'year_normalized', 'is_host', 'years_since_first_participation',
    
    # 国家特征
    'NOC_encoded', 'historical_participation_count', 'historical_medal_rate',
    
    # 交互特征
    'athlete_medal_interaction', 'host_medal_interaction', 'year_medal_interaction',
]

# 检查特征是否存在
available_features = [f for f in feature_cols if f in base.columns]
missing_features = [f for f in feature_cols if f not in base.columns]

if missing_features:
    print(f"  警告: 以下特征缺失: {missing_features}")

print(f"  使用特征数: {len(available_features)}")

# ============================================================================
# 8. 准备训练数据
# ============================================================================
print("\n" + "="*80)
print("准备训练数据")
print("="*80)

# 移除第一年数据（没有历史特征）
base_clean = base[base['Year'] > 1960].copy()

# 移除缺失值过多的行
base_clean = base_clean.dropna(subset=available_features)

# 准备X和y
X = base_clean[available_features]
y = base_clean['has_medal']

print(f"  总样本数: {len(X)}")
print(f"  特征数: {len(available_features)}")
print(f"  正样本（有奖牌）: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  负样本（无奖牌）: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

# 时间分割：训练集（1964-2016），测试集（2024）
train_mask = base_clean['Year'] <= 2016
test_mask = base_clean['Year'] == 2024

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"\n  训练集: {len(X_train)} 样本 (1964-2016)")
print(f"    正样本: {y_train.sum()}, 负样本: {(y_train==0).sum()}")
print(f"  测试集: {len(X_test)} 样本 (2024)")
if len(X_test) > 0:
    print(f"    正样本: {y_test.sum()}, 负样本: {(y_test==0).sum()}")

# ============================================================================
# 9. 训练XGBoost模型
# ============================================================================
print("\n" + "="*80)
print("训练XGBoost模型")
print("="*80)

# 处理类别不平衡
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

print(f"  类别不平衡比例: {scale_pos_weight:.2f}")

# ============================================================================
# 超参数搜索
# ============================================================================
print("\n" + "="*80)
print("超参数搜索")
print("="*80)

# 定义参数搜索空间
param_distributions = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'n_estimators': [100, 200, 300, 400],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
}

# 基础模型
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# 使用RandomizedSearchCV进行搜索
print("  开始超参数搜索（使用RandomizedSearchCV）...")
print(f"  搜索空间: {len(param_distributions)} 个参数")
print(f"  随机搜索次数: 50 次")
print(f"  交叉验证折数: 5 折")

# 使用AUC-ROC作为评分标准
# 直接使用roc_auc_score，StratifiedKFold已经确保每折都有两个类别
scorer = 'roc_auc'  # 使用内置的roc_auc评分，更稳定

# 使用StratifiedKFold确保每折都有两个类别
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,  # 随机搜索50组参数
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True,
    error_score=np.nan  # 如果出错返回nan，继续搜索其他参数
)

# 执行搜索
random_search.fit(X_train, y_train)

print("  超参数搜索完成！")

# 显示最佳参数
print("\n  最佳参数:")
best_params = random_search.best_params_
for param, value in sorted(best_params.items()):
    print(f"    {param}: {value}")

best_score = random_search.best_score_
if pd.notna(best_score):
    print(f"\n  最佳交叉验证 AUC-ROC: {best_score:.4f}")
else:
    print(f"\n  警告: 最佳交叉验证 AUC-ROC 为 nan，可能某些参数组合有问题")
    # 手动计算最佳参数的性能
    best_model = random_search.best_estimator_
    best_score = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
    print(f"  使用最佳参数在训练集上的 AUC-ROC: {best_score:.4f}")
print(f"  最佳参数索引: {random_search.best_index_}")

# 使用最佳参数创建最终模型
print("\n  使用最佳参数训练最终模型...")
model = random_search.best_estimator_

# 保存最佳参数
best_params_file = os.path.join(OUTPUT_DIR, 'best_hyperparameters.json')
with open(best_params_file, 'w', encoding='utf-8') as f:
    json.dump(best_params, f, indent=2, ensure_ascii=False)
print(f"  最佳参数已保存至: out/best_hyperparameters.json")

# 保存搜索结果的详细报告
cv_results_df = pd.DataFrame(random_search.cv_results_)
cv_results_df = cv_results_df.sort_values('mean_test_score', ascending=False)
cv_results_file = os.path.join(OUTPUT_DIR, 'hyperparameter_search_results.csv')
cv_results_df.to_csv(cv_results_file, index=False, encoding='utf-8-sig')
print(f"  搜索结果已保存至: out/hyperparameter_search_results.csv")

# 显示Top 5参数组合（过滤掉nan的结果）
print("\n  Top 5 参数组合:")
# 过滤掉nan的结果
valid_results = cv_results_df[cv_results_df['mean_test_score'].notna()].copy()
if len(valid_results) > 0:
    top5 = valid_results.head(5)
    for rank, (idx, row) in enumerate(top5.iterrows(), 1):
        print(f"\n    排名 {rank}:")
        score = row['mean_test_score']
        std = row['std_test_score'] if pd.notna(row['std_test_score']) else 0
        print(f"      AUC-ROC: {score:.4f} (+/- {std:.4f})")
        params = {k: row[f'param_{k}'] for k in param_distributions.keys()}
        for param, value in sorted(params.items()):
            if pd.notna(value):
                print(f"      {param}: {value}")
else:
    print("    警告: 所有参数组合的评分都是 nan，请检查数据或参数范围")

print("  最终模型训练完成！")

# 诊断：检查是否过拟合
train_accuracy = (model.predict(X_train) == y_train).mean()
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
print(f"\n  诊断信息:")
print(f"    训练集准确率: {train_accuracy:.4f}")
print(f"    训练集AUC-ROC: {train_auc:.4f}")
if train_accuracy > 0.99:
    print(f"    警告: 训练集准确率过高，可能存在过拟合风险")
    print(f"    建议: 检查特征是否只使用历史数据，考虑增加正则化")

# ============================================================================
# 10. 模型评估
# ============================================================================
print("\n" + "="*80)
print("模型评估")
print("="*80)

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# 训练集评估
print("\n【Training Set Performance】")
print(classification_report(y_train, y_train_pred, target_names=['No Medal', 'Has Medal']))
print(f"AUC-ROC: {roc_auc_score(y_train, y_train_proba):.4f}")

# 测试集评估
if len(X_test) > 0 and len(y_test.unique()) > 1:
    print("\n【Test Set Performance (2024)】")
    print(classification_report(y_test, y_test_pred, target_names=['No Medal', 'Has Medal']))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_test_proba):.4f}")
elif len(X_test) > 0:
    print("\n【Test Set Performance (2024)】")
    print(f"  Test set contains only one class, cannot calculate full metrics")
    print(f"  Accuracy: {(y_test == y_test_pred).mean():.4f}")

# ============================================================================
# 11. 特征重要性
# ============================================================================
print("\n" + "="*80)
print("特征重要性 Top 20")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# 保存特征重要性
feature_importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False, encoding='utf-8-sig')
print(f"\n  特征重要性已保存至: out/feature_importance.csv")

# ============================================================================
# 12. 可视化
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

# 2. ROC曲线
ax2 = axes[0, 1]
# 训练集ROC
try:
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    auc_train = roc_auc_score(y_train, y_train_proba)
    ax2.plot(fpr_train, tpr_train, label=f'Train (AUC={auc_train:.3f})')
except:
    pass

# 测试集ROC（如果有两个类别）
if len(X_test) > 0 and len(y_test.unique()) > 1:
    try:
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
        auc_test = roc_auc_score(y_test, y_test_proba)
        ax2.plot(fpr_test, tpr_test, label=f'Test (AUC={auc_test:.3f})')
    except:
        pass

ax2.plot([0, 1], [0, 1], 'k--', label='Random Guess')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 混淆矩阵（测试集）
ax3 = axes[1, 0]
if len(X_test) > 0 and len(y_test.unique()) > 1:
    cm = confusion_matrix(y_test, y_test_pred)
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.figure.colorbar(im, ax=ax3)
    ax3.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=['No Medal', 'Has Medal'],
            yticklabels=['No Medal', 'Has Medal'],
            title='Test Set Confusion Matrix',
            ylabel='True Label',
            xlabel='Predicted Label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
else:
    ax3.text(0.5, 0.5, 'Test set contains\nonly one class', 
            ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Test Set Confusion Matrix')

# 4. 预测概率分布（测试集）
ax4 = axes[1, 1]
if len(X_test) > 0 and len(y_test.unique()) > 1:
    ax4.hist(y_test_proba[y_test == 0], bins=30, alpha=0.5, label='No Medal', color='red')
    ax4.hist(y_test_proba[y_test == 1], bins=30, alpha=0.5, label='Has Medal', color='green')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Test Set Prediction Probability Distribution')
    ax4.legend()
elif len(X_test) > 0:
    ax4.hist(y_test_proba, bins=30, alpha=0.5, color='blue')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Test Set Prediction Probability Distribution (Single Class)')
else:
    ax4.text(0.5, 0.5, 'No test data available', 
            ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Test Set Prediction Probability Distribution')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'xgboost_model_analysis.png'), dpi=300, bbox_inches='tight')
print("  可视化图表已保存至: figure/xgboost_model_analysis.png")

# ============================================================================
# 13. 保存模型和预测结果
# ============================================================================
print("\n" + "="*80)
print("保存模型和结果")
print("="*80)

# 保存模型
import pickle
model_path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  模型已保存至: {os.path.relpath(model_path, PROJECT_ROOT)}")

# 保存预测结果
predictions = base_clean[['NOC', 'Year', 'has_medal']].copy()
predictions['predicted'] = model.predict(X)
predictions['probability'] = model.predict_proba(X)[:, 1]
predictions.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_predictions.csv'), index=False, encoding='utf-8-sig')
print("  预测结果已保存至: out/xgboost_predictions.csv")

# 保存特征列表
with open(os.path.join(OUTPUT_DIR, 'feature_list.txt'), 'w', encoding='utf-8') as f:
    for feat in available_features:
        f.write(f"{feat}\n")
print("  特征列表已保存至: out/feature_list.txt")

print("\n" + "="*80)
print("XGBoost模型构建完成！")
print("="*80)
print("\n生成的文件:")
print(f"  - {os.path.relpath(model_path, PROJECT_ROOT)} (模型文件)")
print("  - out/xgboost_predictions.csv (预测结果)")
print("  - out/feature_importance.csv (特征重要性)")
print("  - figure/xgboost_model_analysis.png (可视化图表)")
print("  - out/feature_list.txt (特征列表)")
