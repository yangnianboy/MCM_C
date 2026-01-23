import pandas as pd
import numpy as np
import os
import warnings
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ===================== 1. 配置根路径 + 通用读取函数 =====================
# 定义数据根路径（原始字符串避免反斜杠转义）
ROOT_DATA_PATH = r"C:\Users\16425\Desktop\MCM_C-main\data\data_dictionary"

def read_olympic_excel(file_name):
    """
    通用Excel读取函数：拼接路径 + 读取 + 验证
    :param file_name: 数据文件名（如"data_dictionary.xls"）
    :return: 读取后的DataFrame
    """
    # 拼接完整路径（os.path.join自动适配Windows/Mac路径分隔符）
    file_path = os.path.join(ROOT_DATA_PATH, file_name)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}，请检查路径/文件名是否正确！")
    
    # 读取Excel（兼容.xls/.xlsx，默认读取第一个sheet）
    try:
        df = pd.read_excel(file_path)
        print(f"✅ 成功导入 {file_name}")
        print(f"   - 数据形状：{df.shape}（行×列）")
        print(f"   - 列名：{df.columns.tolist()[:5]}...（仅展示前5列）\n")  # 避免列名过多刷屏
        return df
    except ImportError:
        # 缺少xlrd库（读取.xls需要）的提示
        raise ImportError("请安装依赖：pip install xlrd openpyxl")
    except Exception as e:
        raise Exception(f"读取失败：{e}")

# ===================== 2. 批量导入所有数据文件 =====================
# 按需求导入（可根据建模需要选择导入哪些文件）
try:
    # 数据字典（用于确认列名）
    df_dict = read_olympic_excel("data_dictionary.xls")
    
    # 奖牌计数数据（核心预测用）
    df_medal_counts = read_olympic_excel("summerOly_medal_counts.xls")
    
    # 运动员数据（分析教练效应用）
    df_athletes = read_olympic_excel("summerOly_athletes.xls")
    
    # 东道主数据（分析东道主效应用）
    df_hosts = read_olympic_excel("summerOly_hosts.xls")
    
    # 赛事项目数据（分析项目与奖牌关联用）
    df_programs = read_olympic_excel("summerOly_programs.xls")

except Exception as e:
    print(f"❌ 数据导入失败：{e}")

# ===================== 3. 数据预处理（适配奖牌预测） =====================
def preprocess_medal_data(df):
    """预处理奖牌数据（基于数据字典调整列名/筛选逻辑）"""
    # 步骤1：根据data_dictionary确认的列名统一字段（示例，需根据实际字典调整）
    column_mapping = {
        "Year": "Year",  # 年份
        "Country_Name": "Country",  # 国家名（根据字典调整）
        "Gold_Medals": "Gold",  # 金牌数
        "Silver_Medals": "Silver",  # 银牌数
        "Bronze_Medals": "Bronze",  # 铜牌数
        "Total_Medals": "Total",  # 总奖牌数
        "Game_Type": "Game_Type"  # 赛事类型（夏季/冬季）
    }
    # 只保留需要的列，并重命名
    df = df.rename(columns=column_mapping)[list(column_mapping.values())]
    
    # 步骤2：筛选夏季奥运会 + 1984年后数据（减少早期异常）
    df = df[df["Game_Type"] == "Summer"]
    df = df[df["Year"] >= 1984]
    
    # 步骤3：缺失值处理 + 国家名标准化
    df = df.dropna(subset=["Country", "Gold", "Total"])  # 核心字段不能为空
    df["Country"] = df["Country"].str.strip()  # 去除首尾空格
    country_mapping = {
        "USA": "United States", "CHN": "China", "JPN": "Japan",
        "AUS": "Australia", "FRA": "France", "NED": "Netherlands", "GBR": "Great Britain"
    }
    df["Country"] = df["Country"].replace(country_mapping)
    
    return df

# 预处理奖牌数据
df_medal_processed = preprocess_medal_data(df_medal_counts)
print(f"✅ 奖牌数据预处理完成，最终数据量：{df_medal_processed.shape}")

# ===================== 4. GM(1,1)模型核心（无需修改） =====================
class GM11:
    def __init__(self, data):
        self.X0 = np.array(data, dtype=float)  # 原始序列
        self.n = len(self.X0)
        self.X1 = self.ago()  # 1-AGO序列
        self.Z1 = self.neighbor_mean()  # 紧邻均值序列
        self.a, self.b = self.fit_params()  # 求解a,b
        self.X0_hat = self.predict_X0()  # 拟合序列
        self.C, self.P = self.posterior_test()  # 模型检验

    def ago(self):
        """一次累加生成"""
        X1 = np.zeros_like(self.X0)
        X1[0] = self.X0[0]
        for i in range(1, self.n):
            X1[i] = X1[i-1] + self.X0[i]
        return X1

    def neighbor_mean(self):
        """紧邻均值生成"""
        Z1 = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            Z1[i] = 0.5 * self.X1[i+1] + 0.5 * self.X1[i]
        return Z1

    def fit_params(self):
        """最小二乘求解a,b"""
        Y = self.X0[1:].reshape(-1, 1)
        B = np.hstack([-self.Z1.reshape(-1, 1), np.ones((self.n-1, 1))])
        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        return params[0, 0], params[1, 0]

    def predict_X1(self, k):
        """预测1-AGO值"""
        return (self.X0[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a if k !=0 else self.X1[0]

    def predict_X0(self):
        """累减还原拟合序列"""
        X0_hat = np.zeros(self.n)
        X0_hat[0] = self.X0[0]
        for k in range(1, self.n):
            X0_hat[k] = self.predict_X1(k) - self.predict_X1(k-1)
        return X0_hat

    def forecast(self, steps):
        """预测未来steps步"""
        forecast = []
        for k in range(self.n, self.n + steps):
            forecast.append(self.predict_X1(k) - self.predict_X1(k-1))
        return np.array(forecast)

    def posterior_test(self):
        """后验差检验"""
        e = self.X0 - self.X0_hat
        mean_X0, var_X0 = np.mean(self.X0), np.var(self.X0)
        mean_e, var_e = np.mean(e), np.var(e)
        C = np.sqrt(var_e) / np.sqrt(var_X0)
        P = np.sum(np.abs(e - mean_e) < 0.6745 * np.sqrt(var_X0)) / self.n
        return C, P

    def get_prediction_interval(self, forecast_vals, alpha=0.05):
        """计算95%预测区间"""
        from scipy.stats import norm
        sigma_e = np.std(self.X0 - self.X0_hat, ddof=1)
        z = norm.ppf(1 - alpha/2)
        lower = np.maximum(forecast_vals - z * sigma_e, 0)
        upper = forecast_vals + z * sigma_e
        return lower, upper

# ===================== 5. 2028奥运奖牌预测（主逻辑） =====================
def predict_2028_medals(df):
    """预测2028洛杉矶奥运会奖牌"""
    target_countries = ["United States", "China", "Japan", "Australia", "France", "Netherlands", "Great Britain"]
    results = []
    forecast_steps = 1  # 预测2028（2024后1步）

    for country in target_countries:
        # 筛选该国数据并按年份排序
        country_df = df[df["Country"] == country].sort_values("Year")
        if len(country_df) < 3:  # 至少3个数据点才建模
            print(f"⚠️ {country} 数据不足，跳过")
            continue

        # 提取金牌/总奖牌序列
        gold_series = country_df["Gold"].values
        total_series = country_df["Total"].values
        gold_2024 = gold_series[-1]
        total_2024 = total_series[-1]

        # 金牌预测
        gm_gold = GM11(gold_series)
        gold_pred = gm_gold.forecast(forecast_steps)[0]
        gold_low, gold_high = gm_gold.get_prediction_interval([gold_pred])

        # 总奖牌预测
        gm_total = GM11(total_series)
        total_pred = gm_total.forecast(forecast_steps)[0]
        total_low, total_high = gm_total.get_prediction_interval([total_pred])

        # 存储结果
        results.append({
            "国家": country,
            "2024金牌": gold_2024,
            "2028金牌预测": round(gold_pred, 1),
            "金牌95%区间": (round(gold_low[0], 1), round(gold_high[0], 1)),
            "金牌模型C/P": (round(gm_gold.C, 3), round(gm_gold.P, 3)),
            "2024总奖牌": total_2024,
            "2028总奖牌预测": round(total_pred, 1),
            "总奖牌95%区间": (round(total_low[0], 1), round(total_high[0], 1)),
            "总奖牌模型C/P": (round(gm_total.C, 3), round(gm_total.P, 3)),
            "金牌趋势": "上升" if gold_pred > gold_2024 else "下降",
            "总奖牌趋势": "上升" if total_pred > total_2024 else "下降"
        })

    # 输出结果
    results_df = pd.DataFrame(results)
    print("\n=== 2028洛杉矶奥运会奖牌预测结果 ===")
    print(results_df)
    return results_df

# 执行预测
pred_results = predict_2028_medals(df_medal_processed)

# ===================== 6. 可视化（中美金牌预测） =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 美国
us_df = df_medal_processed[df_medal_processed["Country"] == "United States"].sort_values("Year")
us_years = us_df["Year"].values
us_gold = us_df["Gold"].values
gm_us = GM11(us_gold)
us_gold_pred = gm_us.forecast(1)[0]
us_low, us_high = gm_us.get_prediction_interval([us_gold_pred])

ax1.plot(us_years, us_gold, "b-o", label="历史金牌数")
ax1.scatter(2028, us_gold_pred, c="red", s=80, label="2028预测值")
ax1.errorbar(2028, us_gold_pred, yerr=[[us_gold_pred - us_low[0]], [us_high[0] - us_gold_pred]], 
             fmt="ro", capsize=5, label="95%预测区间")
ax1.set_title("美国夏季奥运会金牌预测")
ax1.set_xlabel("年份")
ax1.set_ylabel("金牌数")
ax1.legend()
ax1.grid(alpha=0.3)

# 中国
cn_df = df_medal_processed[df_medal_processed["Country"] == "China"].sort_values("Year")
cn_years = cn_df["Year"].values
cn_gold = cn_df["Gold"].values
gm_cn = GM11(cn_gold)
cn_gold_pred = gm_cn.forecast(1)[0]
cn_low, cn_high = gm_cn.get_prediction_interval([cn_gold_pred])

ax2.plot(cn_years, cn_gold, "g-o", label="历史金牌数")
ax2.scatter(2028, cn_gold_pred, c="red", s=80, label="2028预测值")
ax2.errorbar(2028, cn_gold_pred, yerr=[[cn_gold_pred - cn_low[0]], [cn_high[0] - cn_gold_pred]], 
             fmt="ro", capsize=5, label="95%预测区间")
ax2.set_title("中国夏季奥运会金牌预测")
ax2.set_xlabel("年份")
ax2.set_ylabel("金牌数")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()