import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import os
import json
from datetime import datetime

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
                # 兼容旧数据补全机制
                if "到期日" not in df.columns:
                    df["到期日"] = datetime.today().strftime('%Y-%m-%d')
                if "持仓数量(张)" not in df.columns:
                    df["持仓数量(张)"] = 1
                return df
        except:
            pass
    # 默认初始持仓数据
    return pd.DataFrame([
        {"股票代码": "AAPL", "下单行权价(Strike)": 220.0, "收入权利金(Credit)": 1.50, "持仓数量(张)": 2, "到期日": "2026-11-20"},
        {"股票代码": "TSLA", "下单行权价(Strike)": 240.0, "收入权利金(Credit)": 3.80, "持仓数量(张)": 1, "到期日": "2026-11-20"}
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
def estimate_itm_probability_v2(ticker_symbol, strike, expiration_str):
    try:
        from scipy.stats import norm
        
        # 1. 计算剩余天数 (DTE)
        today = datetime.today().date()
        exp_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        
        # 如果已经到期或过期，根据现价直接判断
        ticker = yf.Ticker(ticker_symbol)
        current_price = ticker.fast_info['last_price']
        if dte <= 0:
            return (1.0 if strike > current_price else 0.0), dte
            
        T = dte / 365.0
        r = 0.04 # 假设无风险利率 4%
        
        # 2. 抓取特定到期日的期权链并提取隐含波动率 (IV)
        opt = ticker.option_chain(expiration_str)
        puts = opt.puts
        
        closest_opt = puts.iloc[(puts['strike'] - strike).abs().argsort()[:1]]
        iv = closest_opt['impliedVolatility'].values if not closest_opt.empty else 0.30
        
        if iv == 0 or np.isnan(iv): 
            iv = 0.30 
        
        # 3. Black-Scholes 模型计算被行权率 N(-d2)
        d1 = (np.log(current_price / strike) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        itm_prob = norm.cdf(-d2)
        return itm_prob, dte
        
    except:
        # 降级容错处理
        try:
            today = datetime.today().date()
            exp_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            return None, dte
        except:
            return None, "N/A"

# ================= 3. 侧边栏功能切换菜单与【添加面板】 =================
st.sidebar.header("⚙️ 控制面板")
menu = st.sidebar.selectbox("功能菜单", ["💼 当前持仓管理", "🔍 筛选合适股票"])

st.sidebar.markdown("---")
st.sidebar.subheader("➕ 在此添加新持仓")
with st.sidebar.form(key="add_position_form", clear_on_submit=True):
    new_ticker = st.text_input("股票代码 (如 NVDA)", value="").upper().strip()
    new_strike = st.number_input("下单行权价 (Strike)", min_value=0.0, value=100.0, step=0.5)
    new_credit = st.number_input("收入单张权利金 (Credit)", min_value=0.0, value=1.0, step=0.1)
    new_qty = st.number_input("持仓数量 (张)", min_value=1, value=1, step=1)
    new_exp = st.date_input("期权到期日 (Expiration)", value=datetime.today())
    submit_button = st.form_submit_button(label="确认保存新持仓", use_container_width=True)

if submit_button:
    if new_ticker:
        new_row = pd.DataFrame([{
            "股票代码": new_ticker, 
            "下单行权价(Strike)": new_strike, 
            "收入权利金(Credit)": new_credit,
            "持仓数量(张)": int(new_qty),
            "到期日": new_exp.strftime('%Y-%m-%d')
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
        total_credit_usd = 0.0 
        high_risk_count = 0    
        
        with st.spinner("正在获取实时股价，并结合 DTE 严密精算行权概率..."):
            for index, row in df.iterrows():
                ticker_str = str(row["股票代码"]).upper().strip()
                strike = float(row["下单行权价(Strike)"])
                credit = float(row["收入权利金(Credit)"])
                qty = int(row.get("持仓数量(张)", 1))
                exp_str = str(row.get("到期日", datetime.today().strftime('%Y-%m-%d')))
                
                position_total_credit = credit * qty * 100
                total_credit_usd += position_total_credit
                
                try:
                    t = yf.Ticker(ticker_str)
                    current_price = t.fast_info['last_price']
                    
                    price_diff_pct = ((current_price - strike) / current_price) * 100
                    
                    prob, dte = estimate_itm_probability_v2(ticker_str, strike, exp_str)
                    
                    if prob is not None:
                        prob_val = prob * 100
                        prob_str = f"{prob_val:.1f}%"
                        if prob_val > 50.0:
                            high_risk_count += 1
                    else:
                        prob_str = "无法获取IV"
                    
                    dte_str = f"{dte} 天" if isinstance(dte, int) else str(dte)
                    
                    results.append({
                        "原始索引": index,
                        "股票代码": ticker_str,
                        "持仓数量 (张)": qty,
                        "行权价 (Strike)": strike,
                        "单张权利金 (Credit)": credit,
                        "估计总权利金": f"${position_total_credit:.2f}",
                        "当前正股价 (Current)": round(current_price, 2),
                        "距行权安全垫 (%)": f"{price_diff_pct:.2f}%",
                        "到期日": exp_str,   
                        "剩余天数 (DTE)": dte_str, 
                        "预计被行权概率": prob_str,
                        "勾选删除": False  
                    })
                except:
                    results.append({
                        "原始索引": index, "股票代码": ticker_str, "持仓数量 (张)": qty, "行权价 (Strike)": strike, "单张权利金 (Credit)": credit, "估计总权利金": f"${position_total_credit:.2f}", "当前正股价 (Current)": "获取失败", "距行权安全垫 (%)": "N/A", "到期日": exp_str, "剩余天数 (DTE)": "N/A", "预计被行权概率": "N/A", "勾选删除": False
                    })

        res_df = pd.DataFrame(results)
        
        # 顶部宏观卡片
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(label="💼 运行中的期权单", value=f"{len(df)} 笔")
        with metric_col2:
            st.metric(label="💰 累计锁定权利金", value=f"${total_credit_usd:.2f}")
        with metric_col3:
            st.metric(label="🚨 处于高风险仓位 (>50%行权率)", value=f"{high_risk_count} 笔")
            
        st.markdown("")

        cols = ['勾选删除', '股票代码', '持仓数量 (张)', '行权价 (Strike)', '单张权利金 (Credit)', '估计总权利金', '当前正股价 (Current)', '距行权安全垫 (%)', '到期日', '剩余天数 (DTE)', '预计被行权概率']
        display_df = res_df[cols]
        
        st.subheader("📊 实时持仓监控盘面")
        
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            disabled=['股票代码', '持仓数量 (张)', '行权价 (Strike)', '单张权利金 (Credit)', '估计总权利金', '当前正股价 (Current)', '距行权安全垫 (%)', '到期日', '剩余天数 (DTE)', '预计被行权概率'],
            key="portfolio_editor_v5"
        )
        
        st.markdown("")
        delete_clicked = st.button("🗑️ 删除表格选中持仓", type="primary")
        
        if delete_clicked:
            selected_indices = edited_df[edited_df["勾选删除"] == True].index.tolist()
            
            if selected_indices:
                real_indices_to_drop = [res_df.iloc[i]["原始索引"] for i in selected_indices]
                st.session_state.portfolio_data = st.session_state.portfolio_data.drop(real_indices_to_drop).reset_index(drop=True)
                save_data(st.session_state.portfolio_data)
                st.success(f"成功删除 {len(real_indices_to_drop)} 个持仓项！")
                st.rerun()
            else:
                st.warning("⚠️ 请先在表格第一列中【勾选】您想要删除的股票，然后再点击顶部的『🗑️ 删除表格选中持仓』。")

# ================= 模块二：🔍 筛选合适股票（已全线修复缩进问题） =================
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
