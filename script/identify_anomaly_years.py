import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
athletes_df = pd.read_csv('data/summerOly_athletes.csv')
hosts_df = pd.read_csv('data/summerOly_hosts.csv')

print("="*80)
print("奥运会历史异常年份识别分析")
print("="*80)

# 重点分析1960年后的数据
modern_data = athletes_df[athletes_df['Year'] >= 1960].copy()

print("\n【历史背景下的异常年份】")
print("-"*80)

anomaly_years = {
    1980: {
        'host': '莫斯科 (苏联)',
        'reason': '美国等65个西方国家抵制 (因苏联入侵阿富汗)',
        'impact': '参赛国家和运动员数量大幅减少'
    },
    1984: {
        'host': '洛杉矶 (美国)',
        'reason': '苏联等14个东欧社会主义国家抵制 (报复性抵制)',
        'impact': '东欧强国缺席,奖牌分布异常'
    },
    2020: {
        'host': '东京 (日本)',
        'reason': '新冠疫情推迟至2021年举办',
        'impact': '无观众、严格防疫措施、部分运动员无法参赛'
    }
}

# 分析各年份的参赛情况
print("\n【1960年后各届奥运会参赛数据统计】")
print("-"*80)
print(f"{'年份':<8}{'城市':<20}{'记录数':<10}{'参赛国数':<10}{'获奖国数':<10}{'备注':<30}")
print("-"*80)

yearly_stats = []
for year in sorted(modern_data['Year'].unique()):
    year_data = modern_data[modern_data['Year'] == year]
    
    records = len(year_data)
    num_countries = year_data['NOC'].nunique()
    
    # 获奖国家数
    medal_data = year_data[year_data['Medal'].isin(['Gold', 'Silver', 'Bronze'])]
    num_medal_countries = medal_data['NOC'].nunique()
    
    # 获取城市
    city = year_data['City'].iloc[0] if len(year_data) > 0 else 'Unknown'
    
    note = ""
    if year in anomaly_years:
        note = f"[!] {anomaly_years[year]['reason'][:20]}..."
    
    yearly_stats.append({
        'Year': year,
        'City': city,
        'Records': records,
        'Countries': num_countries,
        'Medal_Countries': num_medal_countries,
        'Note': note
    })
    
    print(f"{year:<8}{city:<20}{records:<10}{num_countries:<10}{num_medal_countries:<10}{note:<30}")

stats_df = pd.DataFrame(yearly_stats)

# 计算统计指标
print("\n" + "="*80)
print("【异常检测：与正常年份对比】")
print("="*80)

# 排除异常年份计算基准线
normal_years = stats_df[~stats_df['Year'].isin(anomaly_years.keys())]
avg_records = normal_years['Records'].mean()
std_records = normal_years['Records'].std()
avg_countries = normal_years['Countries'].mean()
std_countries = normal_years['Countries'].std()

print(f"\n正常年份基准 (排除1980, 1984, 2020):")
print(f"  平均记录数: {avg_records:.0f} ± {std_records:.0f}")
print(f"  平均参赛国: {avg_countries:.0f} ± {std_countries:.0f}")

print("\n异常年份偏离度分析:")
for year in anomaly_years.keys():
    year_stat = stats_df[stats_df['Year'] == year].iloc[0]
    records_dev = (year_stat['Records'] - avg_records) / std_records
    countries_dev = (year_stat['Countries'] - avg_countries) / std_countries
    
    print(f"\n{year}年 ({anomaly_years[year]['host']}):")
    print(f"  历史原因: {anomaly_years[year]['reason']}")
    print(f"  记录数偏离: {records_dev:+.2f} 个标准差 ({year_stat['Records']} vs {avg_records:.0f})")
    print(f"  参赛国偏离: {countries_dev:+.2f} 个标准差 ({year_stat['Countries']} vs {avg_countries:.0f})")
    print(f"  影响评估: {anomaly_years[year]['impact']}")

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 图1: 参赛记录数趋势
ax1 = axes[0]
years = stats_df['Year']
records = stats_df['Records']

ax1.plot(years, records, marker='o', linewidth=2, markersize=6, label='参赛记录数')
ax1.axhline(y=avg_records, color='green', linestyle='--', alpha=0.5, label=f'正常年份均值 ({avg_records:.0f})')
ax1.fill_between(years, avg_records - std_records, avg_records + std_records, 
                  alpha=0.2, color='green', label='±1 标准差范围')

# 标注异常年份
for year in anomaly_years.keys():
    year_stat = stats_df[stats_df['Year'] == year].iloc[0]
    ax1.scatter(year, year_stat['Records'], color='red', s=200, zorder=5, marker='X')
    ax1.annotate(f"{year}\n{anomaly_years[year]['host']}", 
                xy=(year, year_stat['Records']),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax1.set_xlabel('年份', fontsize=12, fontweight='bold')
ax1.set_ylabel('参赛记录数', fontsize=12, fontweight='bold')
ax1.set_title('1960年后夏季奥运会参赛记录数趋势 (异常年份标注)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 参赛国家数趋势
ax2 = axes[1]
countries = stats_df['Countries']

ax2.plot(years, countries, marker='s', linewidth=2, markersize=6, color='orange', label='参赛国家数')
ax2.axhline(y=avg_countries, color='blue', linestyle='--', alpha=0.5, label=f'正常年份均值 ({avg_countries:.0f})')
ax2.fill_between(years, avg_countries - std_countries, avg_countries + std_countries, 
                  alpha=0.2, color='blue', label='±1 标准差范围')

# 标注异常年份
for year in anomaly_years.keys():
    year_stat = stats_df[stats_df['Year'] == year].iloc[0]
    ax2.scatter(year, year_stat['Countries'], color='red', s=200, zorder=5, marker='X')

ax2.set_xlabel('年份', fontsize=12, fontweight='bold')
ax2.set_ylabel('参赛国家数', fontsize=12, fontweight='bold')
ax2.set_title('1960年后夏季奥运会参赛国家数趋势', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hq_data/anomaly_years_analysis.png', dpi=300, bbox_inches='tight')
print("\n\n可视化图表已保存至: hq_data/anomaly_years_analysis.png")

# 生成建议
print("\n" + "="*80)
print("【数据处理建议】")
print("="*80)

print("\n推荐剔除的异常年份及理由:")
print("-"*80)
print("\n1. 强烈建议剔除:")
print("   • 1980年 - 莫斯科奥运会")
print("     理由: 65个西方国家抵制,参赛国家和运动员数量严重失真")
print("     影响: 美国、西德、日本等体育强国缺席")
print("\n   • 1984年 - 洛杉矶奥运会") 
print("     理由: 14个东欧国家抵制,奖牌分布严重异常")
print("     影响: 苏联、东德等传统强国缺席,美国奖牌数异常高")

print("\n2. 根据研究目标决定是否剔除:")
print("   • 2020年 - 东京奥运会 (2021年举办)")
print("     理由: 新冠疫情影响,推迟一年且无观众")
print("     影响: 防疫限制、部分运动员训练受影响")
print("     建议: 如果研究近期趋势可保留,如果研究长期规律建议剔除")

print("\n3. 其他需要注意的年份:")
print("   • 1956年 - 墨尔本奥运会")
print("     注意: 马术项目在斯德哥尔摩单独举办")
print("     建议: 数据相对完整,可以保留")

print("\n" + "="*80)
print("【最终建议的高质量数据范围】")
print("="*80)
print("\n方案一(推荐): 1960-2024年,剔除 1980, 1984, 2020")
print("  • 优点: 数据质量最高,避免政治抵制和疫情影响")
print("  • 适用: 常规趋势分析、预测模型")

print("\n方案二: 1960-2019年,剔除 1980, 1984")
print("  • 优点: 完全避免疫情影响")
print("  • 适用: 需要稳定历史数据的研究")

print("\n方案三: 1988-2024年,剔除 2020")
print("  • 优点: 冷战后时代,全球化、商业化背景一致")
print("  • 适用: 现代奥运会特征研究")

print("\n" + "="*80)

# 保存清洗后的数据示例
print("\n正在生成高质量数据集...")
# 方案一: 剔除1980, 1984, 2020
hq_data_v1 = athletes_df[
    (athletes_df['Year'] >= 1960) & 
    (~athletes_df['Year'].isin([1980, 1984, 2020]))
].copy()

hq_data_v1.to_csv('hq_data/athletes_1960-2024_exclude_1980_1984_2020.csv', index=False)
print(f"✓ 已保存: hq_data/athletes_1960-2024_exclude_1980_1984_2020.csv")
print(f"  记录数: {len(hq_data_v1):,} 条")

# 方案二: 剔除1980, 1984, 截止2019
hq_data_v2 = athletes_df[
    (athletes_df['Year'] >= 1960) & 
    (athletes_df['Year'] <= 2019) &
    (~athletes_df['Year'].isin([1980, 1984]))
].copy()

hq_data_v2.to_csv('hq_data/athletes_1960-2019_exclude_1980_1984.csv', index=False)
print(f"✓ 已保存: hq_data/athletes_1960-2019_exclude_1980_1984.csv")
print(f"  记录数: {len(hq_data_v2):,} 条")

print("\n" + "="*80)
print("分析完成!")
print("="*80)
