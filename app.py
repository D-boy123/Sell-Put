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
                df = pd.DataFrame(data)
                # 兼容旧数据补全机制：如果没有数量列，默认补1
                if "持仓数量(张)" not in df.columns:
                    df["持仓数量(张)"] = 1
                return df
        except:
            pass
    # 默认初始持仓数据（加入数量字段）
    return pd.DataFrame([
        {"股票代码": "AAPL", "下单行权价(Strike)": 220.0, "收入权利金(Credit)": 1.50, "持仓数量(张)": 2},
        {"股票代码": "TSLA", "下单行权价(Strike)": 240.0, "收入权利金(Credit)": 3.80, "持仓数量(张)": 1}
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

# ================= 3. 侧边栏功能切换菜单与【添加面板】 =================
st.sidebar.header("⚙️ 控制面板")
menu = st.sidebar.selectbox("功能菜单", ["💼 当前持仓管理", "🔍 筛选合适股票"])

st.sidebar.markdown("---")
st.sidebar.subheader("➕ 在此添加新持仓")
with st.sidebar.form(key="add_position_form", clear_on_submit=True):
    new_ticker = st.text_input("股票代码 (如 NVDA)", value="").upper().strip()
    new_strike = st.number_input("下单行权价 (Strike)", min_value=0.0, value=100.0, step=0.5)
    new_credit = st.number_input("收入单张权利金 (Credit)", min_value=0.0, value=1.0, step=0.1)
    new_qty = st.number_input("持仓数量 (张)", min_value=1, value=1, step=1) # 🌟 新增数量输入
    submit_button = st.form_submit_button(label="确认保存新持仓", use_container_width=True)

if submit_button:
    if new_ticker:
        new_row = pd.DataFrame([{
            "股票代码": new_ticker, 
            "下单行权价(Strike)": new_strike, 
            "收入权利金(Credit)": new_credit,
            "持仓数量(张)": int(new_qty) # 🌟 保存数量
        }])
        st.session_state.portfolio_data = pd.concat([st.session_state.portfolio_data, new_row], ignore_index=True)
        save_data(st.session_state.portfolio_data)
        st.sidebar.success(f"成功保存 {new_ticker} 并存档！")
        st.rerun()
    else:
        st.sidebar.error("请输入有效的股票代码！")

# ================= 模块一：💼 当前持仓管理（主页） =================
if menu == "💼 当前持仓管理":
    st.title("💼 当前持仓动态透视")
    st.write("勾选表格最左侧并点击下方按钮，可完成移除持仓操作。")
    
    st.markdown("---")

    df = st.session_state.portfolio_data

    if df.empty:
        st.info("目前没有任何持仓数据，请在左侧侧边栏输入并录入新持仓。")
    else:
        results = []
        total_credit_usd = 0.0 # 累计总权利金初始化
        high_risk_count = 0    # 高风险单统计初始化
        
        with st.spinner("正在获取实时股价，并严密推算行权概率..."):
            for index, row in df.iterrows():
                ticker_str = str(row["股票代码"]).upper().strip()
                strike = float(row["下单行权价(Strike)"])
                credit = float(row["收入权利金(Credit)"])
                qty = int(row.get("持仓数量(张)", 1)) # 🌟 读取数量
                
                # 计算这笔单子的真实总收益金（美股1张期权合约=100股正股）
                position_total_credit = credit * qty * 100
                total_credit_usd += position_total_credit
                
                try:
                    t = yf.Ticker(ticker_str)
                    current_price = t.fast_info['last_price']
                    
                    price_diff_pct = ((current_price - strike) / current_price) * 100
                    prob = estimate_itm_probability(ticker_str, strike, is_put=True)
                    
                    if prob is not None:
                        prob_val = prob * 100
                        prob_str = f"{prob_val:.1f}%"
                        if prob_val > 50.0:
                            high_risk_count += 1
                    else:
                        prob_str = "无法估算"
                    
                    results.append({
                        "原始索引": index,
                        "股票代码": ticker_str,
                        "持仓数量 (张)": qty,
                        "行权价 (Strike)": strike,
                        "单张权利金 (Credit)": credit,
                        "估计总权利金": f"${position_total_credit:.2f}",
                        "当前正股价 (Current)": round(current_price, 2),
                        "距行权安全垫 (%)": f"{price_diff_pct:.2f}%",
                        "预计被行权概率": prob_str,
                        "勾选删除": False  
                    })
                except:
                    results.append({
                        "原始索引": index, "股票代码": ticker_str, "持仓数量 (张)": qty, "行权价 (Strike)": strike, "单张权利金 (Credit)": credit, "估计总权利金": f"${position_total_credit:.2f}", "当前正股价 (Current)": "获取失败", "距行权安全垫 (%)": "N/A", "预计被行权概率": "N/A", "勾选删除": False
                    })

        res_df = pd.DataFrame(results)
        
        # --- 🌟 顶部加入数据可视化卡片看板 ---
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(label="💼 运行中的期权单", value=f"{len(df)} 笔")
        with metric_col2:
            st.metric(label="💰 累计锁定权利金", value=f"${total_credit_usd:.2f}")
        with metric_col3:
            st.metric(label="🚨 处于高风险仓位 (>50%行权率)", value=f"{high_risk_count} 笔")
            
        st.markdown("")

        # 调整列顺序，确保“持仓数量”和“估计总权利金”优雅地呆在报表里
        cols = ['勾选删除', '股票代码', '持仓数量 (张)', '行权价 (Strike)', '单张权利金 (Credit)', '估计总权利金', '当前正股价 (Current)', '距行权安全垫 (%)', '预计被行权概率']
        display_df = res_df[cols]
        
        st.subheader("📊 实时持仓监控盘面")
        
        # 渲染数据表格
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            disabled=['股票代码', '持仓数量 (张)', '行权价 (Strike)', '单张权利金 (Credit)', '估计总权利金', '当前正股价 (Current)', '距行权安全垫 (%)', '预计被行权概率'],
            key="portfolio_editor_v4"
        )
        
        # --- 删除按钮置于表格上方紧凑分布 ---
        st.markdown("")
        delete_clicked = st.button("🗑️ 删除表格选中持仓", type="primary")
        
        # 删除触发逻辑
        if delete_clicked:
            selected_indices = edited_df[edited_df["勾选删除"] == True].index.tolist()
            
            if selected_indices:
                real_indices_to_drop = [res_df.iloc[i]["原始索引"] for i in selected_indices]
                st.session_state.portfolio_data = st.session_state.portfolio_data.drop(real_indices_to_drop).reset_index(drop=True)
                save_data(st.session_state.portfolio_data)
                st.success(f"成功删除 {len(real_indices_to_drop)} 个持仓项！")
                st.rerun()
            else:
                st.warning("⚠️ 请先在表格第一列中【勾选】您想要删除的股票，然后再点击『🗑️ 删除表格选中持仓』。")

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

