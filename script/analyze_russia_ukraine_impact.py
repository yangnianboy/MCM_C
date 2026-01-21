import pandas as pd
import numpy as np

# 读取数据
athletes_df = pd.read_csv('data/summerOly_athletes.csv')

print("="*80)
print("俄乌战争对2024年巴黎奥运会影响分析")
print("="*80)

# 分析近几届俄罗斯参赛情况
recent_years = [2008, 2012, 2016, 2020, 2024]

print("\n【俄罗斯参赛情况对比】")
print("-"*80)
print(f"{'年份':<8}{'地点':<15}{'俄罗斯运动员':<12}{'获奖数':<10}{'金牌':<8}{'银牌':<8}{'铜牌':<8}")
print("-"*80)

russia_stats = []
for year in recent_years:
    year_data = athletes_df[athletes_df['Year'] == year]
    
    # 俄罗斯和ROC (2020年使用ROC)
    if year == 2020:
        rus_data = year_data[year_data['NOC'].isin(['RUS', 'ROC'])]
        label = "ROC (受禁赛)"
    elif year == 2024:
        # 2024年可能是中立运动员
        rus_data = year_data[year_data['Team'].str.contains('Russia|Individual Neutral', case=False, na=False)]
        label = "中立运动员"
    else:
        rus_data = year_data[year_data['NOC'] == 'RUS']
        label = "RUS"
    
    # 统计
    num_athletes = len(rus_data)
    gold = len(rus_data[rus_data['Medal'] == 'Gold'])
    silver = len(rus_data[rus_data['Medal'] == 'Silver'])
    bronze = len(rus_data[rus_data['Medal'] == 'Bronze'])
    total_medals = gold + silver + bronze
    
    city = year_data['City'].iloc[0] if len(year_data) > 0 else ''
    
    russia_stats.append({
        'Year': year,
        'City': city,
        'Athletes': num_athletes,
        'Medals': total_medals,
        'Gold': gold,
        'Silver': silver,
        'Bronze': bronze
    })
    
    print(f"{year:<8}{city:<15}{num_athletes:<12}{total_medals:<10}{gold:<8}{silver:<8}{bronze:<8}  ({label})")

print("\n【白俄罗斯参赛情况对比】")
print("-"*80)
print(f"{'年份':<8}{'地点':<15}{'白俄罗斯运动员':<12}{'获奖数':<10}{'金牌':<8}{'银牌':<8}{'铜牌':<8}")
print("-"*80)

for year in recent_years:
    year_data = athletes_df[athletes_df['Year'] == year]
    
    blr_data = year_data[year_data['NOC'] == 'BLR']
    
    num_athletes = len(blr_data)
    gold = len(blr_data[blr_data['Medal'] == 'Gold'])
    silver = len(blr_data[blr_data['Medal'] == 'Silver'])
    bronze = len(blr_data[blr_data['Medal'] == 'Bronze'])
    total_medals = gold + silver + bronze
    
    city = year_data['City'].iloc[0] if len(year_data) > 0 else ''
    
    print(f"{year:<8}{city:<15}{num_athletes:<12}{total_medals:<10}{gold:<8}{silver:<8}{bronze:<8}")

# 总体参赛情况对比
print("\n" + "="*80)
print("【2024年整体数据对比】")
print("="*80)

comparison_years = [2016, 2020, 2024]
print(f"\n{'指标':<20}{'2016里约':<15}{'2020东京':<15}{'2024巴黎':<15}{'变化趋势':<20}")
print("-"*80)

for year in comparison_years:
    year_data = athletes_df[athletes_df['Year'] == year]
    
    if year == 2016:
        rio_records = len(year_data)
        rio_countries = year_data['NOC'].nunique()
        rio_medal_countries = year_data[year_data['Medal'].isin(['Gold', 'Silver', 'Bronze'])]['NOC'].nunique()
    elif year == 2020:
        tokyo_records = len(year_data)
        tokyo_countries = year_data['NOC'].nunique()
        tokyo_medal_countries = year_data[year_data['Medal'].isin(['Gold', 'Silver', 'Bronze'])]['NOC'].nunique()
    else:
        paris_records = len(year_data)
        paris_countries = year_data['NOC'].nunique()
        paris_medal_countries = year_data[year_data['Medal'].isin(['Gold', 'Silver', 'Bronze'])]['NOC'].nunique()

print(f"{'总记录数':<20}{rio_records:<15}{tokyo_records:<15}{paris_records:<15}{paris_records - rio_records:+} vs 2016")
print(f"{'参赛国家/地区数':<20}{rio_countries:<15}{tokyo_countries:<15}{paris_countries:<15}{paris_countries - rio_countries:+} vs 2016")
print(f"{'获奖国家数':<20}{rio_medal_countries:<15}{tokyo_medal_countries:<15}{paris_medal_countries:<15}{paris_medal_countries - rio_medal_countries:+} vs 2016")

# 详细分析2024年的异常
print("\n" + "="*80)
print("【俄乌战争影响评估】")
print("="*80)

print("\n背景信息:")
print("-"*80)
print("1. 俄乌战争: 2022年2月开始")
print("2. 国际体育制裁: 俄罗斯和白俄罗斯被禁止以国家身份参赛")
print("3. 2024巴黎奥运会: 允许部分俄白运动员以'中立运动员'身份参赛")
print("   - 不能使用国旗、国歌、国家队服")
print("   - 参赛人数受到严格限制")
print("   - 不计入国家奖牌榜")

print("\n影响评估:")
print("-"*80)

# 计算俄罗斯历史平均参赛人数
avg_rus_before = np.mean([russia_stats[i]['Athletes'] for i in range(3)])  # 2008, 2012, 2016
rus_2024 = russia_stats[4]['Athletes']  # 2024

print(f"\n1. 俄罗斯参赛规模:")
print(f"   2008-2016年平均: ~{avg_rus_before:.0f} 名运动员")
print(f"   2024年实际: {rus_2024} 名中立运动员")
if rus_2024 < avg_rus_before:
    print(f"   减少: {avg_rus_before - rus_2024:.0f} 人 ({(1-rus_2024/avg_rus_before)*100:.1f}%)")

print(f"\n2. 整体影响:")
paris_vs_tokyo = (paris_records - tokyo_records) / tokyo_records * 100
print(f"   记录数变化: {paris_vs_tokyo:+.1f}% (vs 2020东京)")
print(f"   参赛国数变化: {paris_countries - tokyo_countries:+} 个")

print("\n3. 与1980/1984抵制事件对比:")
print("   1980年莫斯科: 参赛国80个, 记录数7191 (严重异常)")
print("   1984年洛杉矶: 参赛国140个, 记录数9454 (明显异常)")
print(f"   2024年巴黎: 参赛国{paris_countries}个, 记录数{paris_records} (接近正常)")

print("\n" + "="*80)
print("【结论与建议】")
print("="*80)

print("\n2024年巴黎奥运会是否应该剔除?")
print("-"*80)

print("\n不建议剔除的理由:")
print("  1. 数据规模基本正常")
print(f"     - 记录数{paris_records}条, 与2020年({tokyo_records})和2016年({rio_records})相当")
print(f"     - 参赛国{paris_countries}个, 与近两届持平")
print(f"     - 获奖国{paris_medal_countries}个, 与历史水平一致")
print("\n  2. 影响范围有限")
print("     - 主要影响俄罗斯和白俄罗斯两国")
print("     - 其他200+国家/地区正常参赛")
print("     - 不同于1980/1984的大规模多国抵制")
print("\n  3. 中立运动员机制")
print("     - 虽然以中立身份, 但优秀运动员仍可参赛")
print("     - 竞技水平未受显著影响")

print("\n需要注意的情况:")
print("  1. 如果研究国家层面指标")
print("     - 俄罗斯国家队数据不完整")
print("     - 需要特别说明或调整分析方法")
print("\n  2. 如果研究特定项目")
print("     - 某些俄罗斯传统强项(体操、摔跤等)可能受影响")
print("     - 需要项目级别的详细分析")

print("\n最终建议:")
print("-"*80)
print("对于大多数研究: 可以保留2024年数据")
print("  - 整体数据质量良好")
print("  - 异常程度远低于1980/1984")
print("  - 代表了当代奥运会的真实状况")
print("\n如果研究国家竞争力: 需要注意说明")
print("  - 俄罗斯数据特殊处理")
print("  - 或在敏感性分析中剔除2024")
print("\n推荐的最终数据范围:")
print("  主方案: 1960-2024年, 剔除1980, 1984")
print("  替代方案: 1960-2024年, 剔除1980, 1984, 2020 (避免疫情影响)")

print("\n" + "="*80)
