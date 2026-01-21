# 练习题

## 项目结构

```text
MCM_C/
├── 2025_MCM_Problem_C.pdf      # 2025年MCM问题C的原始PDF文件
├── data/                        # 数据目录
│   ├── data_dictionary.csv      # 数据字典（字段说明）
│   ├── summerOly_athletes.csv  # 夏季奥运会运动员数据
│   ├── summerOly_hosts.csv     # 夏季奥运会主办城市数据
│   ├── summerOly_medal_counts.csv  # 夏季奥运会奖牌统计
│   └── summerOly_programs.csv  # 夏季奥运会项目数据
├── paper/                       # 论文目录
│   └── mcmthesis/               # MCM/ICM LaTeX模板
│       ├── code/                # 示例代码
│       │   ├── mcmthesis-matlab1.m
│       │   └── mcmthesis-sudoku.cpp
│       ├── figures/             # 图片资源
│       │   ├── example-image-a.pdf
│       │   ├── example-image-b.pdf
│       │   ├── example-image-c.pdf
│       │   ├── mcmthesis-logo.pdf
│       │   └── qrcodewechat.jpg
│       ├── mcmthesis.cls        # LaTeX模板类文件（核心文件）
│       ├── mcmthesis.dtx        # LaTeX类文件源文件
│       ├── mcmthesis-demo.tex   # 模板示例文档（可参考）
│       ├── mcmthesis-demo.pdf   # 模板示例PDF
│       ├── mcmthesis.pdf        # 模板说明文档
│       └── README.md            # 模板使用说明
├── MCM-ICM master 2025美赛特等奖-C/  # 参考论文目录
│   └── *.pdf                    # 2025年MCM/ICM特等奖论文（参考学习）
├── src/                         # 源代码目录（Python分析脚本、模型代码）
├── Problem.md                   # 问题描述（英文）
├── Problem_cn.md                # 问题描述（中文）
├── todo.md                      # 待办事项清单
├── .gitignore                   # Git忽略文件配置
└── README.md                    # 项目说明文档
```

## 目录说明

- **data/**: 存放所有数据文件，包括运动员、主办城市、奖牌统计和项目数据
  - `data_dictionary.csv`: 数据字典，包含所有数据表的字段说明和示例
  - 其他CSV文件为夏季奥运会相关数据

- **paper/**: 存放论文相关文件
  - `mcmthesis/`: MCM/ICM LaTeX模板目录
    - `mcmthesis.cls`: 核心模板类文件，编译论文时必需
    - `mcmthesis-demo.tex`: 示例文档，可作为论文起点
    - `figures/`: 存放论文图片资源
    - `code/`: 代码排版示例

- **MCM-ICM master 2025美赛特等奖-C/**: 2025年MCM/ICM特等奖参考论文，用于学习写作思路、格式和绘图方法

- **src/**: 存放源代码、分析脚本和模型文件（Python）

- **Problem.md / Problem_cn.md**: 问题描述文档，提供中英文版本

>进行测试式改动在各自分支下改动 test/name,不要直接提交在main分支

## Notice

在自己的test分支下可以随便提交更改,保存你认为有价值的内容,但在合并main分支前(或者直接提交在main)确保内容正确,不冲突,

本项目采取分布式管理,成员有相同的管理权,可查看他人分支工作,不可随意更改他人分支代码

## task

项目所需技术栈:Python +latex

会代码者构建模型,在自己测试过后,讨论,提交,合并

弱代码者,查找相关论文,可以是相关理论,领域,获奖论文,经自己先审阅再提出讨论,学习获奖论文写作思路,格式,流程,绘图
研究latex模板(用paper\mcmthesis下的模板),经测试后提交合并

本项目先确定各成员能力边界,暂不做具体分工
