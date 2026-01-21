"""
根据参赛者数据重新整理获奖国家统计
包括所有参赛国家，未获奖的记为0
"""

import pandas as pd
import numpy as np
import os

print("="*80)
print("根据参赛者数据重新整理获奖国家统计")
print("="*80)

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
PROJECT_ROOT = os.path.normpath(PROJECT_ROOT)
HQ_DATA_DIR = os.path.join(PROJECT_ROOT, 'hq_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'out')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n[1/4] 加载数据...")
athletes = pd.read_csv(os.path.join(HQ_DATA_DIR, 'athletes_medals_unique.csv'))

print(f"  运动员数据: {len(athletes):,} 条记录")
print(f"  年份范围: {athletes['Year'].min()} - {athletes['Year'].max()}")
print(f"  国家数: {athletes['NOC'].nunique()} 个")

# ============================================================================
# 2. 获取所有参赛国家-年份组合
# ============================================================================
print("\n[2/4] 获取所有参赛国家-年份组合...")

# 获取所有参赛国家（包括未获奖的）
all_participants = athletes.groupby(['NOC', 'Year']).size().reset_index(name='athlete_count')
all_participants = all_participants[['NOC', 'Year']].copy()

print(f"  参赛国家-年份组合: {len(all_participants):,} 个")

# ============================================================================
# 3. 统计奖牌数
# ============================================================================
print("\n[3/4] 统计奖牌数...")

# 只保留获奖记录
medals = athletes[athletes['Medal'].isin(['Gold', 'Silver', 'Bronze'])].copy()

# 按国家-年份-奖牌类型统计
medal_counts = medals.groupby(['NOC', 'Year', 'Medal']).size().reset_index(name='count')

# 转换为宽格式
medal_pivot = medal_counts.pivot_table(
    index=['NOC', 'Year'],
    columns='Medal',
    values='count',
    fill_value=0
).reset_index()

# 确保所有列都存在
for medal_type in ['Gold', 'Silver', 'Bronze']:
    if medal_type not in medal_pivot.columns:
        medal_pivot[medal_type] = 0

medal_pivot = medal_pivot[['NOC', 'Year', 'Gold', 'Silver', 'Bronze']]
medal_pivot['Total'] = medal_pivot['Gold'] + medal_pivot['Silver'] + medal_pivot['Bronze']

print(f"  获奖国家-年份组合: {len(medal_pivot):,} 个")

# ============================================================================
# 4. 合并所有参赛国家和奖牌统计
# ============================================================================
print("\n[4/4] 合并数据（包括未获奖国家）...")

# 左连接，保留所有参赛国家
final_counts = all_participants.merge(
    medal_pivot,
    on=['NOC', 'Year'],
    how='left'
)

# 填充缺失值（未获奖的国家）
final_counts['Gold'] = final_counts['Gold'].fillna(0).astype(int)
final_counts['Silver'] = final_counts['Silver'].fillna(0).astype(int)
final_counts['Bronze'] = final_counts['Bronze'].fillna(0).astype(int)
final_counts['Total'] = final_counts['Total'].fillna(0).astype(int)

# 添加排名（按总奖牌数）
final_counts = final_counts.sort_values(['Year', 'Total', 'Gold', 'Silver', 'Bronze'], 
                                         ascending=[True, False, False, False, False])
final_counts['Rank'] = final_counts.groupby('Year')['Total'].rank(
    method='min', ascending=False
).astype(int)

# 重新排列列顺序
final_counts = final_counts[['Rank', 'NOC', 'Gold', 'Silver', 'Bronze', 'Total', 'Year']]

# 按年份和国家排序
final_counts = final_counts.sort_values(['Year', 'Rank']).reset_index(drop=True)

print(f"  最终记录数: {len(final_counts):,} 条")
print(f"  包含年份: {sorted(final_counts['Year'].unique())}")

# 统计信息
print("\n统计信息:")
for year in sorted(final_counts['Year'].unique()):
    year_data = final_counts[final_counts['Year'] == year]
    total_countries = len(year_data)
    medal_countries = len(year_data[year_data['Total'] > 0])
    no_medal_countries = total_countries - medal_countries
    
    print(f"  {year}年: 总参赛国 {total_countries}, 获奖国 {medal_countries}, 未获奖国 {no_medal_countries}")

# ============================================================================
# 5. 保存结果
# ============================================================================
print("\n" + "="*80)
print("保存结果")
print("="*80)

output_file = os.path.join(HQ_DATA_DIR, 'medal_counts_from_athletes.csv')
final_counts.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"[OK] 已保存至: {output_file}")
print(f"  记录数: {len(final_counts):,} 条")

# 同时保存到out目录
output_file2 = os.path.join(OUTPUT_DIR, 'medal_counts_from_athletes.csv')
final_counts.to_csv(output_file2, index=False, encoding='utf-8-sig')
print(f"[OK] 已保存至: {output_file2}")

# 生成统计摘要
summary = final_counts.groupby('Year').agg({
    'NOC': 'count',  # 总参赛国数
    'Total': lambda x: (x > 0).sum()  # 获奖国数
}).reset_index()
summary.columns = ['Year', 'Total_Countries', 'Medal_Countries']
summary['No_Medal_Countries'] = summary['Total_Countries'] - summary['Medal_Countries']

summary_file = os.path.join(OUTPUT_DIR, 'medal_counts_summary.csv')
summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
print(f"[OK] 统计摘要已保存至: {summary_file}")

print("\n" + "="*80)
print("完成！")
print("="*80)
print("\n生成的文件:")
print(f"  - hq_data/medal_counts_from_athletes.csv (完整奖牌统计，包含未获奖国家)")
print(f"  - out/medal_counts_from_athletes.csv (同上)")
print(f"  - out/medal_counts_summary.csv (按年份统计摘要)")
