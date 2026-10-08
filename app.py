import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import os
import json
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Sell Put 策略管理系统", page_icon="📈")

# ================= 1. 本地数据持久化保存机制 =================
DB_FILE = "portfolio.json"

def load_data():
    """从本地 JSON 文件读取持仓数据"""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return pd.DataFrame(data)
        except:
            pass
    # 默认初始数据
    return pd.DataFrame([
        {"股票代码": "AAPL", "下单行权价(Strike)": 220.0, "收入权利金(Credit)": 1.50},
        {"股票代码": "TSLA", "下单行权价(Strike)": 240.0, "收入权利金(Credit)": 3.80}
    ])

def save_data(df):
    """将持仓数据保存到本地 JSON 文件"""
    df_to_save = df.reset_index(drop=True)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(df_to_save.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

# 初始化 Session State
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = load_data()

# ================= 2. 行权概率估算函数 =================
def estimate_itm_probability(ticker_symbol, strike, is_put=True):
    try:
        ticker = yf.Ticker(ticker_symbol)
        current_price = ticker.fast_info['last_price']
        expirations = ticker.options
        if not expirations:
            return 0.5
        
        opt = ticker.option_chain(expirations[0]) # 取最近一个到期日
        calls_or_puts = opt.puts if is_put else opt.calls
        closest_opt = calls_or_puts.iloc[(calls_or_puts['strike'] - strike).abs().argsort()[:1]]
        if closest_opt.empty:
            return 0.5
            
        iv = closest_opt['impliedVolatility'].values[0]
        T = 30 / 365.0 
        r = 0.04 
        if iv == 0 or np.isnan(iv): iv = 0.30 
        
        d1 = (np.log(current_price / strike) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        itm_prob = norm.cdf(-d2) if is_put else norm.cdf(d2)
        return itm_prob
    except:
        return None

# ================= 3. 侧边栏菜单切换 =================
menu = st.sidebar.selectbox("功能菜单", ["主页分析", "当前持仓"])

# ================= 场景一：主页分析 =================
if menu == "主页分析":
    st.title("📈 Sell Put (Short Put) 期权策略分析系统")
    
    # 概览卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📊 已录入持仓标的数", value=f"{len(st.session_state.portfolio_data)} 个")
    with col2:
        total_credit = st.session_state.portfolio_data["收入权利金(Credit)"].sum() if not st.session_state.portfolio_data.empty else 0
        st.metric(label="💰 累计锁定权利金", value=f"${total_credit:.2f}")
    with col3:
        st.metric(label="⏱️ 策略监测状态", value="实时运行中")

    st.markdown("---")
    st.subheader("💡 什么是 Sell Put 策略？")
    st.write(
        "卖出看跌期权（Sell Put）是一种**顺势赚取现金流**或者**折价建仓**的经典期权策略。 "
        "当您卖出一个 Put 时，您有义务在股价跌破行权价时，以该行权价买入股票。作为回报，您立即获得一笔权利金（Credit）。"
    )
    
    st.subheader("🛠️ 核心盈利逻辑")
    st.markdown(
        """
        - **时间价值流逝（Theta 损耗）**：只要股价不大幅下跌，期权价值会随时间流逝而归零，卖方全额稳赚权利金。
        - **股价震荡或上涨**：正股价格上涨或横盘，Put 期权都会变成垃圾归零，策略达成最大利润。
        - **低价抄底**：即便不幸被行权，您也是以『行权价 - 权利金』的更低成本买入心仪的股票，比直接买现货更划算。
        """
    )
    
    st.warning(
        "⚠️ **风险提示**：如果正股遭遇黑天鹅事件暴跌，您必须无条件以行权价接盘，可能产生浮亏。 "
        "请时刻关注『当前持仓』中的**行权安全垫**和**行权概率**，防止穿仓！"
    )

# ================= 场景二：当前持仓 =================
elif menu == "当前持仓":
    st.title("💼 当前持仓管理")
    
    # --- 操作按钮区域：添加新持仓 ---
    if st.button("➕ 添加新持仓（弹窗输入）"):
        st.dialog("add_position_modal") # 触发 Streamlit 的原生弹窗机制
        
    # 定义弹窗内的内容
    @st.dialog("添加新持仓")
    def add_position_modal():
        st.write("请输入您的新期权单数据：")
        new_ticker = st.text_input("股票代码 (如 AAPL, TSLA)", value="").upper().strip()
        new_strike = st.number_input("下单行权价 (Strike Amount)", min_value=0.0, value=100.0, step=0.5)
        new_credit = st.number_input("收入权利金 (Credit Amount)", min_value=0.0, value=1.0, step=0.1)
        
        if st.button("确认保存", use_container_width=True):
            if new_ticker:
                # 组装新行
                new_row = pd.DataFrame([{"股票代码": new_ticker, "下单行权价(Strike)": new_strike, "收入权利金(Credit)": new_credit}])
                # 追加并保存
                st.session_state.portfolio_data = pd.concat([st.session_state.portfolio_data, new_row], ignore_index=True)
                save_data(st.session_state.portfolio_data)
                st.success(f"成功添加 {new_ticker} 持仓！")
                st.rerun() # 刷新页面
            else:
                st.error("请输入有效的股票代码！")

    st.markdown("---")

    # --- 核心逻辑：数据读取与展示 ---
    df = st.session_state.portfolio_data

    if df.empty:
        st.info("目前没有持仓数据，请点击上方按钮添加。")
    else:
        # 实时抓取行情并计算衍生指标
        results = []
        with st.spinner("正在获取雅虎财经实时数据并计算行权概率..."):
            for index, row in df.iterrows():
                ticker_str = str(row["股票代码"]).upper().strip()
                strike = float(row["下单行权价(Strike)"])
                credit = float(row["收入权利金(Credit)"])
                
                try:
                    t = yf.Ticker(ticker_str)
                    current_price = t.fast_info['last_price']
                    
                    # 计算安全垫差异 (%)
                    price_diff_pct = ((current_price - strike) / current_price) * 100
                    
                    # 估算行权概率
                    prob = estimate_itm_probability(ticker_str, strike, is_put=True)
                    prob_str = f"{prob*100:.1f}%" if prob is not None else "无法估算"
                    
                    results.append({
                        "ID": index, # 用于删除标识
                        "股票代码": ticker_str,
                        "行权价 (Strike)": strike,
                        "权利金 (Credit)": credit,
                        "当前估价 (Current)": round(current_price, 2),
                        "距行权安全垫 (%)": f"{price_diff_pct:.2f}%",
                        "预计被行权概率": prob_str
                    })
                except:
                    results.append({
                        "ID": index,
                        "股票代码": ticker_str,
                        "行权价 (Strike)": strike,
                        "权利金 (Credit)": credit,
                        "当前估价 (Current)": "获取失败",
                        "距行权安全垫 (%)": "N/A",
                        "预计被行权概率": "N/A"
                    })

        # 显示精美结果表格 (非编辑模式)
        res_df = pd.DataFrame(results)
        display_df = res_df.drop(columns=["ID"]) # 隐藏不美观的ID
        st.subheader("📊 实时持仓透视表")
        st.dataframe(display_df, use_container_width=True)

        # --- 操作区域：单条持仓删除功能 ---
        st.markdown("---")
        st.subheader("🗑️ 删除持仓")
        delete_list = [f"{r['股票代码']} (Strike: {r['行权价 (Strike)']})" for r in results]
        selected_to_delete = st.selectbox("选择要删除的持仓项：", delete_list)
        
        if st.button("🔥 确认删除选中持仓", type="primary"):
            # 找到选中项对应的原始索引号
            selected_index = delete_list.index(selected_to_delete)
            target_id = results[selected_index]["ID"]
            
            # 从原始 Session State 中剔除并保存
            st.session_state.portfolio_data = st.session_state.portfolio_data.drop(target_id).reset_index(drop=True)
            save_data(st.session_state.portfolio_data)
            st.success("持仓已成功删除！数据已同步到本地保存。")
            st.rerun()
