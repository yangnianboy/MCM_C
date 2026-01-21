# 奥运会高质量数据集 - 数据清洗报告
生成时间: 2026-01-22 02:00:43

## 1. 数据范围
- 时间范围: 1960-2024年
- 剔除年份: 1980年(莫斯科), 1984年(洛杉矶), 2020年(东京)
- 包含届次: 14 届奥运会
- 年份列表: [np.int64(1960), np.int64(1964), np.int64(1968), np.int64(1972), np.int64(1976), np.int64(1988), np.int64(1992), np.int64(1996), np.int64(2000), np.int64(2004), np.int64(2008), np.int64(2012), np.int64(2016), np.int64(2024)]

## 2. 数据规模
- 原始数据: 252,565 条记录
- 清洗后: 164,514 条记录
- 去重后奖牌数据: 117,509 条记录
- 参赛国家/地区: 224 个
- 运动项目: 58 个

## 3. 数据清洗内容
### 3.1 国家/地区名称标准化
- 应用映射: 8 个
- 主要变更:
  - GDR -> GER
  - FRG -> GER
  - ROC -> RUS
  - RU1 -> RUS

### 3.2 城市名称统一
- 应用映射: 7 个
  - Athina -> Athens
  - Roma -> Rome
  - Moskva -> Moscow
  - München -> Munich
  - Montréal -> Montreal
  - Ciudad de México -> Mexico City
  - Antwerpen -> Antwerp

### 3.3 团体项目处理
- 团体项目记录: 57,438 条
- 个人项目记录: 107,076 条
- 说明: 团体项目已标记，去重数据集中每个团队只计数一次

### 3.4 特殊实体分类
- Regular: 164,409 条
- Individual/Independent: 93 条
- Refugee Team: 12 条

### 3.5 项目变化
- 已停止项目记录: 0 条
- 新增项目记录: 877 条

## 4. 数据文件说明
### 4.1 athletes_cleaned_full.csv
- 完整的运动员级别数据
- 包含所有清洗和标记字段
- 团体项目未去重（每个队员一条记录）
- 适用场景: 运动员级别分析、参赛情况分析

### 4.2 athletes_medals_unique.csv
- 去重后的数据
- 团体项目每个队伍只保留一条记录
- 适用场景: 奖牌统计、国家排名分析

### 4.3 medal_counts_calculated.csv
- 从运动员数据计算的奖牌统计
- 按年份和国家汇总
- 适用场景: 与官方数据对比验证

### 4.4 medal_counts_official_cleaned.csv
- 清洗后的官方奖牌统计数据
- 应用了国家代码标准化
- 适用场景: 国家层面的奖牌分析

## 5. 新增字段说明
| 字段名 | 说明 |
|--------|------|
| NOC_original | 原始国家代码 |
| City_original | 原始城市名称 |
| is_team_event | 是否为团体项目 (True/False) |
| entity_type | 实体类型 (Regular/Mixed Team/Refugee Team/Individual/Independent) |
| is_discontinued_sport | 是否为已停止项目 (True/False) |
| is_new_sport | 是否为新增项目 (True/False) |

## 6. 使用建议
### 6.1 奖牌统计分析
- 推荐使用: `athletes_medals_unique.csv` 或 `medal_counts_official_cleaned.csv`
- 原因: 避免团体项目重复计数

### 6.2 运动员参赛分析
- 推荐使用: `athletes_cleaned_full.csv`
- 原因: 保留所有运动员记录

### 6.3 国家历史分析
- 注意: 使用 `NOC_original` 字段可追溯国家变更历史
- 示例: 区分东德(GDR)、西德(FRG)、统一后德国(GER)

### 6.4 特殊情况处理
- 过滤特殊实体: 使用 `entity_type=='Regular'` 只保留常规参赛队伍
- 筛选特定类型: 使用 `entity_type=='Mixed Team'` 等筛选特殊实体
- 已停止项目: 可通过 `is_discontinued_sport=False` 过滤

## 7. 数据质量评估
- 时间一致性: [OK] 统一使用1960-2024年范围，剔除异常年份
- 国家代码一致性: [OK] 已标准化处理
- 城市名称一致性: [OK] 已统一处理
- 团体项目处理: [OK] 已标记并提供去重版本
- 特殊实体标记: [OK] 已标记
- 项目变化标记: [OK] 已标记
