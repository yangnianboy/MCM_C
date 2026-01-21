import pandas as pd
import numpy as np

# 读取运动员数据
df = pd.read_csv('data/summerOly_athletes.csv')

print("="*60)
print("奥运会数据质量分析")
print("="*60)
print(f"\n数据总行数: {len(df):,}")
print(f"数据列数: {len(df.columns)}")
print(f"年份范围: {df['Year'].min()} - {df['Year'].max()}")

print("\n" + "="*60)
print("各年份数据量统计")
print("="*60)
year_counts = df['Year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"{year}: {count:,} 条记录")

print("\n" + "="*60)
print("各年份数据完整性分析")
print("="*60)

results = []
for year in sorted(df['Year'].unique()):
    subset = df[df['Year']==year]
    total_cells = len(subset) * len(subset.columns)
    missing_cells = subset.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    
    # 检查各列的缺失情况
    name_missing = subset['Name'].isnull().sum()
    sex_missing = subset['Sex'].isnull().sum()
    team_missing = subset['Team'].isnull().sum()
    
    results.append({
        'Year': year,
        'Records': len(subset),
        'Missing_Pct': missing_pct,
        'Name_Missing': name_missing,
        'Sex_Missing': sex_missing,
        'Team_Missing': team_missing
    })
    
    print(f"{year}: 记录数={len(subset):>5}, "
          f"整体缺失率={missing_pct:.2f}%, "
          f"姓名缺失={name_missing}, "
          f"性别缺失={sex_missing}, "
          f"队伍缺失={team_missing}")

# 识别数据质量显著改善的年份
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("数据质量分析总结")
print("="*60)

# 1896-1920年代早期奥运会
early_data = results_df[results_df['Year'] <= 1920]
print(f"\n1896-1920年 (早期): 平均记录数 = {early_data['Records'].mean():.0f}")

# 1920-1960年代
mid_data = results_df[(results_df['Year'] > 1920) & (results_df['Year'] <= 1960)]
print(f"1920-1960年 (发展期): 平均记录数 = {mid_data['Records'].mean():.0f}")

# 1960年后
modern_data = results_df[results_df['Year'] > 1960]
print(f"1960年后 (现代): 平均记录数 = {modern_data['Records'].mean():.0f}")

print("\n" + "="*60)
print("关键年份节点分析")
print("="*60)

# 找出记录数显著增加的年份
for i in range(1, len(results_df)):
    prev_records = results_df.iloc[i-1]['Records']
    curr_records = results_df.iloc[i]['Records']
    if curr_records > prev_records * 1.5:  # 增长超过50%
        print(f"{results_df.iloc[i]['Year']}: 记录数大幅增加 ({prev_records} -> {curr_records})")

# 1960年作为一个重要的分界点
print("\n推荐分析起始年份:")
print("- 1960年及以后: 现代奥运会，数据记录更完整、系统化")
print("- 1896-1960年: 早期数据，可能存在记录不完整的情况")
