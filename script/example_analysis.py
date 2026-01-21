"""
高质量数据集使用示例
演示常见的分析场景
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("高质量奥运会数据集 - 使用示例")
print("="*80)

# ============================================================================
# 示例1: 奖牌统计分析（推荐使用去重数据）
# ============================================================================
print("\n【示例1】奖牌统计分析")
print("-"*80)

# 加载去重后的数据
df = pd.read_csv('hq_data/athletes_medals_unique.csv')

# 只保留获奖记录
medals = df[df['Medal'].isin(['Gold', 'Silver', 'Bronze'])]

# 统计2024年奖牌榜前10名
medals_2024 = medals[medals['Year'] == 2024]
medal_summary = medals_2024.groupby('NOC')['Medal'].value_counts().unstack(fill_value=0)
medal_summary['Total'] = medal_summary.sum(axis=1)
medal_summary = medal_summary.sort_values('Total', ascending=False).head(10)

print("\n2024年巴黎奥运会奖牌榜 (Top 10):")
print(medal_summary[['Gold', 'Silver', 'Bronze', 'Total']])

# ============================================================================
# 示例2: 历史趋势分析
# ============================================================================
print("\n【示例2】历史趋势分析")
print("-"*80)

# 统计各届奥运会的参赛规模
yearly_stats = df.groupby('Year').agg({
    'NOC': 'nunique',  # 参赛国家数
    'Name': 'count'     # 总记录数
}).rename(columns={'NOC': 'Countries', 'Name': 'Records'})

print("\n各届奥运会参赛规模:")
print(yearly_stats)

# ============================================================================
# 示例3: 国家竞争力对比
# ============================================================================
print("\n【示例3】国家竞争力对比")
print("-"*80)

# 对比中美两国的历年奖牌数
countries = ['USA', 'CHN']
comparison = []

for year in sorted(medals['Year'].unique()):
    for noc in countries:
        year_noc_medals = medals[(medals['Year']==year) & (medals['NOC']==noc)]
        if len(year_noc_medals) > 0:
            gold = len(year_noc_medals[year_noc_medals['Medal']=='Gold'])
            total = len(year_noc_medals)
            comparison.append({'Year': year, 'NOC': noc, 'Gold': gold, 'Total': total})

comparison_df = pd.DataFrame(comparison)
print("\n美国vs中国历年奖牌数对比:")
print(comparison_df.pivot_table(index='Year', columns='NOC', values='Total', fill_value=0))

# ============================================================================
# 示例4: 性别平等分析
# ============================================================================
print("\n【示例4】性别平等分析")
print("-"*80)

# 使用完整数据分析性别比例
df_full = pd.read_csv('hq_data/athletes_cleaned_full.csv')

gender_by_year = df_full.groupby(['Year', 'Sex']).size().unstack()
gender_by_year['Female_Pct'] = gender_by_year['F'] / (gender_by_year['F'] + gender_by_year['M']) * 100

print("\n各届奥运会女性运动员比例:")
print(gender_by_year[['M', 'F', 'Female_Pct']])

# ============================================================================
# 示例5: 团体项目vs个人项目分析
# ============================================================================
print("\n【示例5】团体项目vs个人项目分析")
print("-"*80)

team_stats = df_full.groupby(['Year', 'is_team_event']).size().unstack()
team_stats.columns = ['Individual', 'Team']
team_stats['Team_Pct'] = team_stats['Team'] / (team_stats['Team'] + team_stats['Individual']) * 100

print("\n各届奥运会团体项目占比:")
print(team_stats)

# ============================================================================
# 示例6: 特定国家历史分析（含国家变更）
# ============================================================================
print("\n【示例6】德国历史分析（含东德、西德）")
print("-"*80)

germany_data = df_full[df_full['NOC'] == 'GER']

# 按原始国家代码分组
germany_history = germany_data.groupby(['Year', 'NOC_original']).size().unstack(fill_value=0)

print("\n德国参赛记录（区分东德GDR、西德FRG、统一后GER）:")
print(germany_history)

# ============================================================================
# 示例7: 新增项目分析
# ============================================================================
print("\n【示例7】新增项目分析")
print("-"*80)

new_sports = df_full[df_full['is_new_sport'] == True]
new_sports_summary = new_sports.groupby(['Year', 'Sport']).size().unstack(fill_value=0)

print("\n新增项目参赛情况:")
print(new_sports_summary)

# ============================================================================
# 示例8: 使用官方奖牌数据
# ============================================================================
print("\n【示例8】使用官方奖牌统计")
print("-"*80)

medal_official = pd.read_csv('hq_data/medal_counts_official_cleaned.csv')

# 找出各届奥运会的金牌榜冠军
champions = medal_official.loc[medal_official.groupby('Year')['Gold'].idxmax()]
print("\n各届奥运会金牌榜冠军:")
print(champions[['Year', 'NOC', 'Gold', 'Silver', 'Bronze', 'Total']])

print("\n" + "="*80)
print("示例演示完成！")
print("="*80)
print("\n[9] 实体类型分析")
print("-"*80)

entity_stats = df_full['entity_type'].value_counts()
print("\n实体类型分布:")
for entity_type, count in entity_stats.items():
    pct = count / len(df_full) * 100
    print(f"  - {entity_type}: {count:,} 条 ({pct:.2f}%)")

# 过滤示例
regular_only = df_full[df_full['entity_type'] == 'Regular']
print(f"\n仅保留常规参赛: {len(regular_only):,} 条记录")

print("\n" + "="*80)
print("提示:")
print("1. 奖牌统计推荐使用: athletes_medals_unique.csv")
print("2. 运动员分析推荐使用: athletes_cleaned_full.csv")
print("3. 国家层面分析可使用: medal_counts_official_cleaned.csv")
print("4. 注意区分团体项目和个人项目")
print("5. 使用NOC_original字段可追溯国家历史变更")
print("6. 使用entity_type字段可过滤或分析特殊参赛实体")
