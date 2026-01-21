"""
奥运会数据清洗脚本
解决数据质量问题，生成高质量数据集
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("奥运会数据清洗 - 创建高质量数据集")
print("="*80)

# ============================================================================
# 1. 加载原始数据
# ============================================================================
print("\n[1/7] 加载原始数据...")

# 尝试不同编码加载数据
def load_csv_with_encoding(filepath):
    """尝试不同编码加载CSV文件"""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 如果都失败，使用latin1（几乎总能成功）
    return pd.read_csv(filepath, encoding='latin1')

athletes_df = load_csv_with_encoding('data/summerOly_athletes.csv')
medal_counts_df = load_csv_with_encoding('data/summerOly_medal_counts.csv')
hosts_df = load_csv_with_encoding('data/summerOly_hosts.csv')
programs_df = load_csv_with_encoding('data/summerOly_programs.csv')

print(f"  athletes: {len(athletes_df):,} 条记录")
print(f"  medal_counts: {len(medal_counts_df):,} 条记录")
print(f"  hosts: {len(hosts_df):,} 条记录")
print(f"  programs: {len(programs_df):,} 条记录")

# ============================================================================
# 2. 时间范围和异常年份过滤
# ============================================================================
print("\n[2/7] 应用时间范围和异常年份过滤...")
print("  选择: 1960年及以后")
print("  剔除: 1980年(莫斯科，西方抵制), 1984年(洛杉矶，东欧抵制), 2020年(东京，疫情)")

# 定义有效年份
ANOMALY_YEARS = [1980, 1984, 2020]
valid_years = [y for y in range(1960, 2025, 4) if y not in ANOMALY_YEARS]
print(f"  有效年份: {valid_years}")

# 过滤数据
athletes_clean = athletes_df[
    (athletes_df['Year'] >= 1960) & 
    (~athletes_df['Year'].isin(ANOMALY_YEARS))
].copy()

medal_counts_clean = medal_counts_df[
    (medal_counts_df['Year'] >= 1960) & 
    (~medal_counts_df['Year'].isin(ANOMALY_YEARS))
].copy()

print(f"  过滤后 athletes: {len(athletes_clean):,} 条记录 (保留 {len(athletes_clean)/len(athletes_df)*100:.1f}%)")
print(f"  过滤后 medal_counts: {len(medal_counts_clean):,} 条记录")

# ============================================================================
# 3. 国家/地区名称变更问题 - 建立映射表
# ============================================================================
print("\n[3/7] 处理国家/地区名称变更问题...")

# 国家代码标准化映射
COUNTRY_MAPPING = {
    # 德国统一
    'GDR': 'GER',  # 东德 -> 德国
    'FRG': 'GER',  # 西德 -> 德国
    
    # 苏联解体后保留原代码（便于历史分析）
    'URS': 'URS',  # 苏联保持原样
    'EUN': 'EUN',  # 独联体保持原样
    
    # 南斯拉夫解体保留原代码
    'YUG': 'YUG',  # 南斯拉夫保持原样
    
    # 捷克斯洛伐克分裂保留原代码
    'TCH': 'TCH',  # 捷克斯洛伐克保持原样
    
    # 其他历史变更
    'ROC': 'RUS',  # 俄罗斯奥委会 -> 俄罗斯
    'RU1': 'RUS',  # 俄罗斯变体
}

# 应用映射
athletes_clean['NOC_original'] = athletes_clean['NOC']
athletes_clean['NOC'] = athletes_clean['NOC'].replace(COUNTRY_MAPPING)

# 特殊实体标记 - 改为分类字段
def classify_entity(team_name):
    """将特殊实体分类"""
    if pd.isna(team_name):
        return 'Regular'
    
    team_str = str(team_name)
    
    # 检查各类特殊实体
    if 'Mixed' in team_str:
        return 'Mixed Team'
    elif 'Refugee' in team_str:
        return 'Refugee Team'
    elif 'Individual Olympic' in team_str or 'Independent Olympic' in team_str:
        return 'Individual/Independent'
    else:
        return 'Regular'

athletes_clean['entity_type'] = athletes_clean['Team'].apply(classify_entity)

print(f"  标准化国家代码: {len(COUNTRY_MAPPING)} 个映射")
print(f"  实体分类统计:")
for entity_type, count in athletes_clean['entity_type'].value_counts().items():
    print(f"    - {entity_type}: {count:,} 条")

# 对medal_counts也应用映射
medal_counts_clean['NOC_original'] = medal_counts_clean['NOC']
medal_counts_clean['NOC'] = medal_counts_clean['NOC'].replace(COUNTRY_MAPPING)

# ============================================================================
# 4. 城市名称统一
# ============================================================================
print("\n[4/7] 统一城市名称...")

CITY_MAPPING = {
    'Athina': 'Athens',
    'Roma': 'Rome',
    'Moskva': 'Moscow',
    'München': 'Munich',
    'Montréal': 'Montreal',
    'Ciudad de México': 'Mexico City',
    'Antwerpen': 'Antwerp',
}

athletes_clean['City_original'] = athletes_clean['City']
athletes_clean['City'] = athletes_clean['City'].replace(CITY_MAPPING)

print(f"  统一城市名称: {len(CITY_MAPPING)} 个映射")

# ============================================================================
# 5. 处理团体项目重复计数问题
# ============================================================================
print("\n[5/7] 识别团体项目...")

# 团体项目关键词
TEAM_SPORTS_KEYWORDS = [
    'Basketball', 'Football', 'Handball', 'Hockey', 'Volleyball',
    'Water Polo', 'Rugby', 'Baseball', 'Softball'
]

TEAM_EVENTS_KEYWORDS = [
    'Team', 'Relay', '4 x', 'Doubles', 'Pairs', 'Quadruple',
    'Coxed', 'Coxless', 'Eight'
]

def is_team_event(sport, event):
    """判断是否为团体项目"""
    if pd.isna(sport) or pd.isna(event):
        return False
    
    sport_str = str(sport)
    event_str = str(event)
    
    # 检查团体运动
    for keyword in TEAM_SPORTS_KEYWORDS:
        if keyword in sport_str:
            return True
    
    # 检查团体项目关键词
    for keyword in TEAM_EVENTS_KEYWORDS:
        if keyword in event_str:
            return True
    
    return False

athletes_clean['is_team_event'] = athletes_clean.apply(
    lambda row: is_team_event(row['Sport'], row['Event']), 
    axis=1
)

print(f"  团体项目记录: {athletes_clean['is_team_event'].sum():,} 条")
print(f"  个人项目记录: {(~athletes_clean['is_team_event']).sum():,} 条")

# 创建去重后的奖牌统计
# 团体项目：每个(Year, NOC, Event, Medal)只计数一次
def create_unique_medals(df):
    """创建去重后的奖牌数据"""
    # 个人项目直接使用
    individual_medals = df[~df['is_team_event']].copy()
    
    # 团体项目去重
    team_medals = df[df['is_team_event']].copy()
    team_medals_unique = team_medals.drop_duplicates(
        subset=['Year', 'NOC', 'Event', 'Medal']
    )
    
    print(f"    团体项目去重: {len(team_medals):,} -> {len(team_medals_unique):,} 条")
    
    return pd.concat([individual_medals, team_medals_unique])

# ============================================================================
# 6. 项目变化标记
# ============================================================================
print("\n[6/7] 标记项目变化...")

# 已停止的项目
DISCONTINUED_SPORTS = [
    'Art Competitions', 'Tug-Of-War', 'Cricket', 'Croquet', 
    'Polo', 'Rackets', 'Roque', 'Jeu de Paume', 'Basque Pelota',
    'Water Motorsports'
]

# 新增项目（2000年后）
NEW_SPORTS = [
    'Breaking', 'Skateboarding', 'Sport Climbing', 'Surfing',
    'Karate', 'Triathlon'
]

athletes_clean['is_discontinued_sport'] = athletes_clean['Sport'].isin(DISCONTINUED_SPORTS)
athletes_clean['is_new_sport'] = athletes_clean['Sport'].isin(NEW_SPORTS)

print(f"  已停止项目记录: {athletes_clean['is_discontinued_sport'].sum():,} 条")
print(f"  新增项目记录: {athletes_clean['is_new_sport'].sum():,} 条")

# ============================================================================
# 7. 数据验证 - athletes vs medal_counts
# ============================================================================
print("\n[7/7] 验证数据一致性...")

# 从athletes计算奖牌数
def calculate_medals_from_athletes(df):
    """从运动员数据计算国家奖牌数"""
    medals_only = df[df['Medal'].isin(['Gold', 'Silver', 'Bronze'])].copy()
    
    # 对团体项目去重
    team_medals = medals_only[medals_only['is_team_event']].drop_duplicates(
        subset=['Year', 'NOC', 'Event', 'Medal']
    )
    individual_medals = medals_only[~medals_only['is_team_event']]
    
    all_medals = pd.concat([team_medals, individual_medals])
    
    # 按年份和国家统计
    medal_summary = all_medals.groupby(['Year', 'NOC', 'Medal']).size().unstack(fill_value=0)
    medal_summary = medal_summary.reset_index()
    
    if 'Gold' not in medal_summary.columns:
        medal_summary['Gold'] = 0
    if 'Silver' not in medal_summary.columns:
        medal_summary['Silver'] = 0
    if 'Bronze' not in medal_summary.columns:
        medal_summary['Bronze'] = 0
    
    medal_summary['Total'] = medal_summary['Gold'] + medal_summary['Silver'] + medal_summary['Bronze']
    
    return medal_summary

athletes_medal_calc = calculate_medals_from_athletes(athletes_clean)

# 对比验证（抽样检查几个国家）
print("\n  样本对比 (去重后的athletes计算 vs official medal_counts):")
print("  " + "-"*70)

sample_years = [1960, 1988, 2000, 2016, 2024]
sample_countries = ['USA', 'CHN', 'GBR', 'RUS', 'GER']

for year in sample_years:
    if year in athletes_clean['Year'].values:
        for noc in sample_countries:
            athletes_data = athletes_medal_calc[
                (athletes_medal_calc['Year'] == year) & 
                (athletes_medal_calc['NOC'] == noc)
            ]
            
            official_data = medal_counts_clean[
                (medal_counts_clean['Year'] == year) & 
                (medal_counts_clean['NOC'] == noc)
            ]
            
            if len(athletes_data) > 0 and len(official_data) > 0:
                athletes_total = athletes_data['Total'].values[0]
                official_total = official_data['Total'].values[0]
                diff = athletes_total - official_total
                
                if abs(diff) > 5:  # 只显示差异较大的
                    print(f"  {year} {noc}: Athletes计算={athletes_total}, Official={official_total}, 差异={diff:+d}")

# ============================================================================
# 保存清洗后的数据
# ============================================================================
print("\n" + "="*80)
print("保存高质量数据集...")
print("="*80)

# 保存完整的清洗后数据
output_file = 'hq_data/athletes_cleaned_full.csv'
athletes_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"[OK] {output_file}")
print(f"  记录数: {len(athletes_clean):,}")

# 保存去重后的奖牌数据（用于奖牌统计）
medals_unique = create_unique_medals(athletes_clean)
output_file = 'hq_data/athletes_medals_unique.csv'
medals_unique.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"[OK] {output_file}")
print(f"  记录数: {len(medals_unique):,}")

# 保存计算的奖牌统计
output_file = 'hq_data/medal_counts_calculated.csv'
athletes_medal_calc.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"[OK] {output_file}")
print(f"  记录数: {len(athletes_medal_calc):,}")

# 保存清洗后的官方奖牌数据
output_file = 'hq_data/medal_counts_official_cleaned.csv'
medal_counts_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"[OK] {output_file}")
print(f"  记录数: {len(medal_counts_clean):,}")

# ============================================================================
# 生成数据质量报告
# ============================================================================
print("\n" + "="*80)
print("生成数据质量报告...")
print("="*80)

report = []
report.append("# 奥运会高质量数据集 - 数据清洗报告\n")
report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append("\n## 1. 数据范围\n")
report.append(f"- 时间范围: 1960-2024年\n")
report.append(f"- 剔除年份: 1980年(莫斯科), 1984年(洛杉矶), 2020年(东京)\n")
report.append(f"- 包含届次: {len(athletes_clean['Year'].unique())} 届奥运会\n")
report.append(f"- 年份列表: {sorted(athletes_clean['Year'].unique())}\n")

report.append("\n## 2. 数据规模\n")
report.append(f"- 原始数据: {len(athletes_df):,} 条记录\n")
report.append(f"- 清洗后: {len(athletes_clean):,} 条记录\n")
report.append(f"- 去重后奖牌数据: {len(medals_unique):,} 条记录\n")
report.append(f"- 参赛国家/地区: {athletes_clean['NOC'].nunique()} 个\n")
report.append(f"- 运动项目: {athletes_clean['Sport'].nunique()} 个\n")

report.append("\n## 3. 数据清洗内容\n")
report.append("### 3.1 国家/地区名称标准化\n")
report.append(f"- 应用映射: {len(COUNTRY_MAPPING)} 个\n")
report.append("- 主要变更:\n")
for old, new in COUNTRY_MAPPING.items():
    if old != new:
        report.append(f"  - {old} -> {new}\n")

report.append("\n### 3.2 城市名称统一\n")
report.append(f"- 应用映射: {len(CITY_MAPPING)} 个\n")
for old, new in CITY_MAPPING.items():
    report.append(f"  - {old} -> {new}\n")

report.append("\n### 3.3 团体项目处理\n")
report.append(f"- 团体项目记录: {athletes_clean['is_team_event'].sum():,} 条\n")
report.append(f"- 个人项目记录: {(~athletes_clean['is_team_event']).sum():,} 条\n")
report.append("- 说明: 团体项目已标记，去重数据集中每个团队只计数一次\n")

report.append("\n### 3.4 特殊实体分类\n")
entity_stats = athletes_clean['entity_type'].value_counts()
for entity_type, count in entity_stats.items():
    report.append(f"- {entity_type}: {count:,} 条\n")

report.append("\n### 3.5 项目变化\n")
report.append(f"- 已停止项目记录: {athletes_clean['is_discontinued_sport'].sum():,} 条\n")
report.append(f"- 新增项目记录: {athletes_clean['is_new_sport'].sum():,} 条\n")

report.append("\n## 4. 数据文件说明\n")
report.append("### 4.1 athletes_cleaned_full.csv\n")
report.append("- 完整的运动员级别数据\n")
report.append("- 包含所有清洗和标记字段\n")
report.append("- 团体项目未去重（每个队员一条记录）\n")
report.append("- 适用场景: 运动员级别分析、参赛情况分析\n")

report.append("\n### 4.2 athletes_medals_unique.csv\n")
report.append("- 去重后的数据\n")
report.append("- 团体项目每个队伍只保留一条记录\n")
report.append("- 适用场景: 奖牌统计、国家排名分析\n")

report.append("\n### 4.3 medal_counts_calculated.csv\n")
report.append("- 从运动员数据计算的奖牌统计\n")
report.append("- 按年份和国家汇总\n")
report.append("- 适用场景: 与官方数据对比验证\n")

report.append("\n### 4.4 medal_counts_official_cleaned.csv\n")
report.append("- 清洗后的官方奖牌统计数据\n")
report.append("- 应用了国家代码标准化\n")
report.append("- 适用场景: 国家层面的奖牌分析\n")

report.append("\n## 5. 新增字段说明\n")
report.append("| 字段名 | 说明 |\n")
report.append("|--------|------|\n")
report.append("| NOC_original | 原始国家代码 |\n")
report.append("| City_original | 原始城市名称 |\n")
report.append("| is_team_event | 是否为团体项目 (True/False) |\n")
report.append("| entity_type | 实体类型 (Regular/Mixed Team/Refugee Team/Individual/Independent) |\n")
report.append("| is_discontinued_sport | 是否为已停止项目 (True/False) |\n")
report.append("| is_new_sport | 是否为新增项目 (True/False) |\n")

report.append("\n## 6. 使用建议\n")
report.append("### 6.1 奖牌统计分析\n")
report.append("- 推荐使用: `athletes_medals_unique.csv` 或 `medal_counts_official_cleaned.csv`\n")
report.append("- 原因: 避免团体项目重复计数\n")

report.append("\n### 6.2 运动员参赛分析\n")
report.append("- 推荐使用: `athletes_cleaned_full.csv`\n")
report.append("- 原因: 保留所有运动员记录\n")

report.append("\n### 6.3 国家历史分析\n")
report.append("- 注意: 使用 `NOC_original` 字段可追溯国家变更历史\n")
report.append("- 示例: 区分东德(GDR)、西德(FRG)、统一后德国(GER)\n")

report.append("\n### 6.4 特殊情况处理\n")
report.append("- 过滤特殊实体: 使用 `entity_type=='Regular'` 只保留常规参赛队伍\n")
report.append("- 筛选特定类型: 使用 `entity_type=='Mixed Team'` 等筛选特殊实体\n")
report.append("- 已停止项目: 可通过 `is_discontinued_sport=False` 过滤\n")

report.append("\n## 7. 数据质量评估\n")
report.append("- 时间一致性: [OK] 统一使用1960-2024年范围，剔除异常年份\n")
report.append("- 国家代码一致性: [OK] 已标准化处理\n")
report.append("- 城市名称一致性: [OK] 已统一处理\n")
report.append("- 团体项目处理: [OK] 已标记并提供去重版本\n")
report.append("- 特殊实体标记: [OK] 已标记\n")
report.append("- 项目变化标记: [OK] 已标记\n")

# 保存报告
report_file = 'hq_data/DATA_CLEANING_REPORT.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.writelines(report)

print(f"[OK] {report_file}")

print("\n" + "="*80)
print("数据清洗完成！")
print("="*80)
print("\n高质量数据集已生成在 hq_data/ 目录下")
print("请查看 DATA_CLEANING_REPORT.md 了解详细信息")
