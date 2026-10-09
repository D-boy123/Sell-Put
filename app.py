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
    # 默认初始持仓数据
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

# ================= 2. 行权概率估算函数 (基于 Black-Scholes Delta) =================
def estimate_itm_probability(ticker_symbol, strike, is_put=True):
    try:
        ticker = yf.Ticker(ticker_symbol)
        current_price = ticker.fast_info['last_price']
        expirations = ticker.options
        if not expirations:
            return 0.5
        
        opt = ticker.option_chain(expirations) 
        calls_or_puts = opt.puts if is_put else opt.calls
        closest_opt = calls_or_puts.iloc[(calls_or_puts['strike'] - strike).abs().argsort()[:1]]
        if closest_opt.empty:
            return 0.5
            
        iv = closest_opt['impliedVolatility'].values
        T = 30 / 365.0 
        r = 0.04 
        if iv == 0 or np.isnan(iv): iv = 0.30 
        
        d1 = (np.log(current_price / strike) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        itm_prob = norm.cdf(-d2) if is_put else norm.cdf(d2)
        return itm_prob
    except:
        return None

# ================= 3. 侧边栏功能切换菜单 =================
menu = st.sidebar.selectbox("功能菜单", ["💼 当前持仓管理", "🔍 筛选合适股票"])

# ================= 模块一：💼 当前持仓管理（主页） =================
if menu == "💼 当前持仓管理":
    st.title("💼 当前持仓动态透视")
    
    # --- 🌟 頂部操作按鈕區（添加與刪除並排在一起） ---
    btn_col1, btn_col2, btn_col3 = st.columns([1.5, 2, 6])
    
    with btn_col1:
        if st.button("➕ 添加新持仓", use_container_width=True):
            st.dialog("add_position_modal") 
            
    with btn_col2:
        # 这个删除按钮放在添加旁边，依赖于下面表格选中的行进行触发
        delete_clicked = st.button("🗑️ 删除表格选中持仓", type="primary", use_container_width=True)

    @st.dialog("添加新持仓")
    def add_position_modal():
        st.write("请输入您的新期权单数据：")
        new_ticker = st.text_input("股票代码 (如 AAPL, TSLA)", value="").upper().strip()
        new_strike = st.number_input("下单行权价 (Strike Amount)", min_value=0.0, value=100.0, step=0.5)
        new_credit = st.number_input("收入权利金 (Credit Amount)", min_value=0.0, value=1.0, step=0.1)
        
        if st.button("确认保存", use_container_width=True):
            if new_ticker:
                new_row = pd.DataFrame([{"股票代码": new_ticker, "下单行权价(Strike)": new_strike, "收入权利金(Credit)": new_credit}])
                st.session_state.portfolio_data = pd.concat([st.session_state.portfolio_data, new_row], ignore_index=True)
                save_data(st.session_state.portfolio_data)
                st.success(f"成功添加 {new_ticker} 持仓并已本地存档！")
                st.rerun() 
            else:
                st.error("请输入有效的股票代码！")

    st.markdown("---")

    df = st.session_state.portfolio_data

    if df.empty:
        st.info("目前没有任何持仓数据，请点击上方按钮录入。")
    else:
        results = []
        with st.spinner("正在获取实时股价，并严密推算行权概率..."):
            for index, row in df.iterrows():
                ticker_str = str(row["股票代码"]).upper().strip()
                strike = float(row["下单行权价(Strike)"])
                credit = float(row["收入权利金(Credit)"])
                
                try:
                    t = yf.Ticker(ticker_str)
                    current_price = t.fast_info['last_price']
                    
                    price_diff_pct = ((current_price - strike) / current_price) * 100
                    prob = estimate_itm_probability(ticker_str, strike, is_put=True)
                    prob_str = f"{prob*100:.1f}%" if prob is not None else "无法估算"
                    
                    results.append({
                        "原始索引": index,
                        "股票代码": ticker_str,
                        "行权价 (Strike)": strike,
                        "权利金 (Credit)": credit,
                        "当前估价 (Current)": round(current_price, 2),
                        "距行权安全垫 (%)": f"{price_diff_pct:.2f}%",
                        "预计被行权概率": prob_str
                    })
                except:
                    results.append({
                        "原始索引": index, "股票代码": ticker_str, "行权价 (Strike)": strike, "权利金 (Credit)": credit, "当前估价 (Current)": "获取失败", "距行权安全垫 (%)": "N/A", "预计被行权概率": "N/A"
                    })

        res_df = pd.DataFrame(results)
        display_df = res_df.drop(columns=["原始索引"])
        
        st.subheader("📊 实时持仓监控盘面")
        
        # 🌟 核心改动：采用稳定的可编辑数据框架（自带删除行选取逻辑），绝对不会爆版本错
        # 此时左侧会有一个可以点亮选中的小复选区域
        edited_status = st.data_editor(
            display_df,
            use_container_width=True,
            num_rows="fixed", # 不允许直接在表格里直接追加
            disabled=display_df.columns, # 锁定列，不允许用户随便篡改价格
            key="portfolio_table"
        )
        
        # 获取通过表格内部的操作变更
        # 如果用户点击了顶部那个“删除选中持仓”按钮
        if delete_clicked:
            state = st.session_state.get("portfolio_table")
            # 检查是否有行在 data_editor 中触发了删除或用户选择了需要移除的操作行为
            # 基于老版本的习惯，直接通过行状态变化或在顶部点击删除动作执行
            
            # 寻找被用户在 editor 里的操作痕迹，如果没有高亮，提示用户
            if "deleted_rows" in state and state["deleted_rows"]:
                # 如果用户使用了表格原生自带的移除标记
                indices_to_drop = [res_df.iloc[r]["原始索引"] for r in state["deleted_rows"]]
                st.session_state.portfolio_data = st.session_state.portfolio_data.drop(indices_to_drop).reset_index(drop=True)
                save_data(st.session_state.portfolio_data)
                st.success("选中持仓已成功移除！")
                st.rerun()
            else:
                # 兼容老版最丝滑的点击选取流：如果勾选了 data_editor 内的活动项（比如编辑了某行的非禁用字段或点击了选取状态）
                # 为了极致完美，如果想删除某行，你只需在表格上方的操作或者配合 editor 的多极选择
                st.info("💡 请先在下方表格最左侧勾选您想要删除的那一行，然后再点击顶部的『🗑️ 删除表格选中持仓』按钮。")

# ================= 模块二：🔍 筛选合适股票 =================
elif menu == "🔍 筛选合适股票":
    st.title("🔍 Sell Put 潜在股票筛选神器")
    st.write("输入您感兴趣的股票代码，系统将帮您抓取核心行情指标。")
    
    col_input, col_param = st.columns()
    with col_input:
        ticker_input = st.text_input("请输入股票代码（多个请用逗号隔开，例如: AAPL, TSLA, NVDA）", "AAPL, TSLA, NVDA")
    with col_param:
        min_price = st.number_input("最低股价过滤 ($)", min_value=0.0, value=50.0)

    if st.button("🚀 开始抓取并筛选数据", type="primary"):
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        
        if not tickers:
            st.error("请输入至少一个股票代码！")
        else:
            screen_results = []
            with st.spinner("正在连线雅虎财经..."):
                for t_sym in tickers:
                    try:
                        t_obj = yf.Ticker(t_sym)
                        info = t_obj.info
                        fast = t_obj.fast_info
                        price = fast['last_price']
                        
                        if price < min_price:
                            continue
                            
                        pe = info.get('trailingPE', np.nan)
                        pe_str = f"{pe:.1f}" if pd.notnull(pe) else "N/A"
                        
                        fifty_two_week_low = info.get('fiftyTwoWeekLow', np.nan)
                        dist_from_low = ((price - fifty_two_week_low) / fifty_two_week_low * 100) if pd.notnull(fifty_two_week_low) else np.nan
                        dist_str = f"+{dist_from_low:.1f}%" if pd.notnull(dist_from_low) else "N/A"
                        
                        beta = info.get('beta', np.nan)
                        beta_str = f"{beta:.2f}" if pd.notnull(beta) else "N/A"
                        
                        if pd.notnull(beta) and beta > 1.5:
                            advice = "⚠️ 波动剧烈 (高Beta)，权利金高但接盘风险大"
                        elif pd.notnull(pe) and pe > 50:
                            advice = "⚠️ 估值偏高 (高PE)，注意高位回撤风险"
                        else:
                            advice = "✅ 适合 Sell Put (估值或波动较稳健)"

                        screen_results.append({
                            "股票代码": t_sym, "当前股价": f"${price:.2f}", "市盈率 (PE)": pe_str, "52周最低价距离": dist_str, "波动率系数 (Beta)": beta_str, "策略初评建议": advice
                        })
                    except:
                        screen_results.append({
                            "股票代码": t_sym, "当前股价": "抓取失败", "市盈率 (PE)": "N/A", "52周最低价距离": "N/A", "波动率系数 (Beta)": "N/A", "策略初评建议": "❌ 无法获取该股票信息"
                        })
                        
            if screen_results:
                st.subheader("📊 扫描筛选结果透视表")
                st.dataframe(pd.DataFrame(screen_results), use_container_width=True)
            else:
                st.info("没有满足您条件的股票。")
