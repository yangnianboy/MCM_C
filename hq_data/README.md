# 高质量奥运会数据集

经过系统清洗的1960-2024年夏季奥运会数据（剔除1980、1984、2020异常年份）。

## 📁 数据文件（选择指南）

| 文件名 | 记录数 | 用途 | 适用场景 |
|--------|--------|------|---------|
| **athletes_medals_unique.csv** ⭐ | 117,509 | 去重奖牌数据 | 奖牌统计、国家排名 |
| athletes_cleaned_full.csv | 164,514 | 完整运动员数据 | 参赛分析、性别研究 |
| medal_counts_official_cleaned.csv | 917 | 官方奖牌汇总 | 国家层面分析 |
| medal_counts_calculated.csv | 912 | 计算奖牌汇总 | 数据验证 |

**推荐使用**: `athletes_medals_unique.csv` (团体项目已去重)

---

## 🔧 数据清洗说明

### 1️⃣ 时间筛选
- ✅ 保留: 1960-2024年（14届）
- ❌ 剔除: 1980年(西方抵制)、1984年(东欧抵制)、2020年(疫情)

### 2️⃣ 核心处理
- **国家代码**: 统一东德/西德→GER，保留原始代码可追溯
- **城市名称**: 统一拼写（Roma→Rome等）
- **团体项目**: 提供去重版本（每队一条记录）
- **实体分类**: Regular / Mixed Team / Refugee Team / Individual

### 3️⃣ 新增字段
| 字段 | 说明 |
|------|------|
| `NOC_original` | 原始国家代码（追溯历史） |
| `is_team_event` | 团体项目标记 |
| `entity_type` | 实体类型（4类） |

---

## 💻 快速上手

### 奖牌统计（推荐）
```python
import pandas as pd

# 使用去重数据
df = pd.read_csv('hq_data/athletes_medals_unique.csv')
medals = df[df['Medal'].isin(['Gold', 'Silver', 'Bronze'])]

# 统计各国奖牌
medal_counts = medals.groupby(['Year', 'NOC', 'Medal']).size()
```

### 常用过滤
```python
# 只保留常规国家队
df_regular = df[df['entity_type'] == 'Regular']

# 只保留个人项目
df_individual = df[df['is_team_event'] == False]

# 区分东德/西德/统一德国
ger = df[df['NOC'] == 'GER']
east_ger = ger[ger['NOC_original'] == 'GDR']  # 东德
west_ger = ger[ger['NOC_original'] == 'FRG']  # 西德
```

---

## ⚠️ 重要提示

1. **团体项目**: 奖牌统计必须用 `unique` 版本，否则会重复计数
2. **实体过滤**: 建议使用 `entity_type=='Regular'` 排除特殊实体
3. **数据差异**: `calculated` 和 `official` 奖牌数可能有±2枚差异（正常）

---

## 📊 数据概览

- **时间跨度**: 1960-2024年（14届）
- **数据规模**: 164,514条记录
- **参赛国家**: 224个
- **运动项目**: 58个
- **数据完整性**: ✅ 无缺失值

**Top 5参赛国**: 🇺🇸USA (9,384) · 🇩🇪GER (9,188) · 🇦🇺AUS (6,355) · 🇬🇧GBR (6,069) · 🇫🇷FRA (6,016)

---

## 📖 更多信息

- `DATA_CLEANING_REPORT.md` - 详细清洗报告
- `script/example_analysis.py` - 完整代码示例
- `anomaly_years_analysis.png` - 异常年份可视化

---

**数据质量保证**: 已验证，可用于学术研究和数据分析 | 生成时间: 2026-01-22
