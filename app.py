import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm

st.set_page_config(layout="wide")

# ================= 侧边栏菜单切换 =================
menu = st.sidebar.selectbox("功能菜单", ["主页分析", "当前持仓"])

# ================= 理论计算：行权概率 (基于 Black-Scholes Delta) =================
def estimate_itm_probability(ticker_symbol, strike, is_put=True):
    """
    通过 yfinance 获取最接近的期权链，利用 Delta 或隐含波动率估算 Sell Put 的行权概率
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # 获取现价
        current_price = ticker.fast_info['last_price']
        
        # 获取期权到期日
        expirations = ticker.options
        if not expirations:
            return 0.5 # 无期权数据时返回默认值
        
        # 默认取最近的一个到期日进行估算（实际应用中可根据持仓调整）
        opt = ticker.option_chain(expirations[0])
        calls_or_puts = opt.puts if is_put else opt.calls
        
        # 寻找最接近 Strike 的期权数据以获取隐含波动率(IV)
        closest_opt = calls_or_puts.iloc[(calls_or_puts['strike'] - strike).abs().argsort()[:1]]
        if closest_opt.empty:
            return 0.5
            
        iv = closest_opt['impliedVolatility'].values[0]
        
        # 简单的 Black-Scholes T=30天(0.08年) 估算散点 Delta 作为行权概率近似
        T = 30 / 365.0 
        r = 0.04 # 假设无风险利率 4%
        if iv == 0: iv = 0.30 # 缺省波动率
        
        d1 = (np.log(current_price / strike) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        
        # 对于 Sell Put，ITM(被行权) 的概率是 P(S < K) = N(-d2)
        itm_prob = norm.cdf(-d2) if is_put else norm.cdf(d2)
        return itm_prob
    except:
        return None

# ================= 场景一：主页分析 =================
if menu == "主页分析":
    st.title("📈 欢迎使用 Sell Put 策略分析系统")
    st.write("请在左侧菜单切换到 **当前持仓** 来管理您的仓位。")

# ================= 场景二：当前持仓 =================
elif menu == "当前持仓":
    st.title("💼 当前持仓管理")
    st.write("您可以在下表中**直接双击修改**、**在最下方空白行新增**持仓，或选中行按 Delete 删除。")

    # 初始化本地临时缓存数据（实际生产中可结合数据库或st.session_state）
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame([
            {"股票代码": "AAPL", "下单行权价(Strike)": 220.0, "收入权利金(Credit)": 1.50},
            {"股票代码": "TSLA", "下单行权价(Strike)": 240.0, "收入权利金(Credit)": 3.80}
        ])

    # 使用 Streamlit 强大的可编辑表格展现，允许用户自由增删改
    edited_df = st.data_editor(
        st.session_state.portfolio_data,
        num_rows="dynamic", # 允许动态增加行
        use_container_width=True,
        key="portfolio_editor"
    )
    
    # 保存用户的修改
    st.session_state.portfolio_data = edited_df

    # 当用户点击计算按钮时，自动抓取实时行情并计算衍生指标
    if st.button("🔄 刷新并计算最新状态"):
        with st.spinner("正在获取雅虎财经实时数据及计算行权概率..."):
            results = []
            
            for index, row in edited_df.iterrows():
                ticker_str = str(row["股票代码"]).upper().strip()
                strike = float(row["下单行权价(Strike)"])
                credit = float(row["收入权利金(Credit)"])
                
                if not ticker_str:
                    continue
                    
                try:
                    # 1. 抓取当前价格
                    t = yf.Ticker(ticker_str)
                    current_price = t.fast_info['last_price']
                    
                    # 2. 计算与行权价的差异 (%) 
                    # 对于 Sell Put，现价高于行权价安全。差异 = (现价 - 行权价) / 现价
                    price_diff_pct = ((current_price - strike) / current_price) * 100
                    
                    # 3. 估算行权概率
                    prob = estimate_itm_probability(ticker_str, strike, is_put=True)
                    prob_str = f"{prob*100:.1f}%" if prob is not None else "无法估算"
                    
                    results.append({
                        "股票代码": ticker_str,
                        "行权价 (Strike)": f"${strike:.2f}",
                        "权利金 (Credit)": f"${credit:.2f}",
                        "当前估价 (Current)": f"${current_price:.2f}",
                        "距行权安全垫 (%)": f"{price_diff_pct:.2f}%",
                        "预计被行权概率": prob_str
                    })
                except Exception as e:
                    results.append({
                        "股票代码": ticker_str,
                        "行权价 (Strike)": f"${strike:.2f}",
                        "权利金 (Credit)": f"${credit:.2f}",
                        "当前估价 (Current)": "获取失败",
                        "距行权安全垫 (%)": "N/A",
                        "预计被行权概率": "N/A"
                    })
            
            # 渲染最终的计算结果报表
            if results:
                st.subheader("📊 实时持仓透视表")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)
                
                # 额外的小贴士
                st.info("💡 **指标说明**：\n"
                        "- **距行权安全垫 (%)**：正数代表当前股价高于行权价（安全）；负数代表股价已跌破行权价（将被行权）。\n"
                        "- **预计被行权概率**：基于隐含波动率(IV)与标准期权定价模型(Delta)估算的 30 天内变成实值(ITM)的概率。")
