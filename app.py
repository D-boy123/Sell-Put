import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import os
import json
from datetime import datetime
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
                # 兼容升级旧数据机制
                if "持仓数量(张)" not in df.columns:
                    df["持仓数量(张)"] = 1
                if "到期日" not in df.columns:
                    df["到期日"] = datetime.today().strftime('%Y-%m-%d')
                return df
        except:
            pass
    # 默认初始持仓数据（已完美拓展加入数量与到期日结构）
    return pd.DataFrame([
        {"股票代码": "AAPL", "下单行权阶(Strike)": 220.0, "收入权利金(Credit)": 1.50, "持仓数量(张)": 2, "到期日": "2026-11-20"},
        {"股票代码": "TSLA", "下单行权阶(Strike)": 240.0, "收入权利金(Credit)": 3.80, "持仓数量(张)": 1, "到期日": "2026-11-20"}
    ])

def save_data(df):
    """将持仓数据保存到本地 JSON 文件"""
    df_to_save = df.reset_index(drop=True)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(df_to_save.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

# 初始化 Session State
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = load_data()

# ================= 2. 行权概率估算函数 (基于 Black-Scholes Delta 及特定到期日) =================
def estimate_itm_probability_v3(ticker_symbol, strike, expiration_str):
    try:
        # 计算剩余天数 (DTE)
        today = datetime.today().date()
        exp_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        
        ticker = yf.Ticker(ticker_symbol)
        current_price = ticker.fast_info['last_price']
        
        # 已经到期或过期时的边界直接判断
        if dte <= 0:
            return (1.0 if strike > current_price else 0.0), dte
            
        T = dte / 365.0 
        r = 0.04 
        
        # 精确抓取用户输入的特定到期日期权链
        opt = ticker.option_chain(expiration_str)
        puts = opt.puts
        
        closest_opt = puts.iloc[(puts['strike'] - strike).abs().argsort()[:1]]
        iv = closest_opt['impliedVolatility'].values if not closest_opt.empty else 0.30
        if iv == 0 or np.isnan(iv): 
            iv = 0.30 
        
        d1 = (np.log(current_price / strike) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        itm_prob = norm.cdf(-d2)
        return itm_prob, dte
    except:
        try:
            today = datetime.today().date()
            exp_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            return None, dte
        except:
            return None, "N/A"

# ================= 3. 侧边栏功能切换菜单 =================
menu = st.sidebar.selectbox("功能菜单", ["💼 当前持仓管理", "🔍 筛选合适股票"])

# ================= 模块一：💼 当前持仓管理（主页） =================
if menu == "💼 当前持仓管理":
    st.title("💼 当前持仓动态透视")
    
    # --- 顶栏操作区：添加新持仓 ---
    if st.button("➕ 添加新持仓"):
        st.dialog("add_position_modal") 
        
    @st.dialog("add_position_modal")
    def add_position_modal():
        st.write("请输入您的新期权单数据：")
        new_ticker = st.text_input("股票代码 (如 AAPL, TSLA)", value="").upper().strip()
        new_strike = st.number_input("下单行权价 (Strike Amount)", min_value=0.0, value=100.0, step=0.5)
        new_credit = st.number_input("收入单张权利金 (Credit Amount)", min_value=0.0, value=1.0, step=0.1)
        new_qty = st.number_input("持仓数量 (张)", min_value=1, value=1, step=1)
        new_exp = st.date_input("期权到期日 (Expiration Date)", value=datetime.today())
        
        if st.button("确认保存", use_container_width=True):
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
        total_credit_usd = 0.0 # 汇总初始化
        
        with st.spinner("正在获取实时股价，并严密推算行权概率..."):
            for index, row in df.iterrows():
                ticker_str = str(row["股票代码"]).upper().strip()
                strike = float(row.get("下单行权价(Strike)", row.get("下单行权价(Strike)", 100.0)))
                # 兼容旧版本键名转换
                if "下单行权价(Strike)" not in row:
                    strike = float(row.get("下单行权价(Strike)", 100.0))
                credit = float(row["收入权利金(Credit)"])
                qty = int(row.get("持仓数量(张)", 1))
                exp_str = str(row.get("到期日", datetime.today().strftime('%Y-%m-%d')))
                
                # 美股每张标准期权合约 = 100股正股
                position_total_credit = credit * qty * 100
                total_credit_usd += position_total_credit
                
                try:
                    t = yf.Ticker(ticker_str)
                    current_price = t.fast_info['last_price']
                    
                    price_diff_pct = ((current_price - strike) / current_price) * 100
                    prob, dte = estimate_itm_probability_v3(ticker_str, strike, exp_str)
                    prob_str = f"{prob*100:.1f}%" if prob is not None else "无法估算"
                    dte_str = f"{dte} 天" if isinstance(dte, int) else str(dte)
                    
                    results.append({
                        "原始索引": index,
                        "股票代码": ticker_str,
                        "数量 (张)": qty,
                        "行权价 (Strike)": strike,
                        "单张权利金": credit,
                        "估计总权利金": f"${position_total_credit:.2f}",
                        "当前正股价 (Current)": round(current_price, 2),
                        "距行权安全垫 (%)": f"{price_diff_pct:.2f}%",
                        "到期日": exp_str,
                        "剩余天数 (DTE)": dte_str,
                        "预计被行权概率": prob_str
                    })
                except:
                    results.append({
                        "原始索引": index, "股票代码": ticker_str, "数量 (张)": qty, "行权价 (Strike)": strike, "单张权利金": credit, "估计总权利金": f"${position_total_credit:.2f}", "当前正股价 (Current)": "获取失败", "距行权安全垫 (%)": "N/A", "到期日": exp_str, "剩余天数 (DTE)": "N/A", "预计被行权概率": "N/A"
                    })

        res_df = pd.DataFrame(results)
        
        # 顶端新增一个一目了然的宏观锁定金额看板
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric(label="💼 当前追踪期权单", value=f"{len(df)} 笔")
        with col_metric2:
            st.metric(label="💰 累计安全落袋总权利金", value=f"${total_credit_usd:.2f}")
            
        st.markdown("")
        display_df = res_df.drop(columns=["原始索引"])
        
        st.subheader("📊 实时持仓监控盘面")
        st.dataframe(display_df, use_container_width=True)
        
        # --- 下方保留您一贯最习惯、100%无损兼容的稳定删除区 ---
        st.markdown("---")
        del_col1, del_col2 = st.columns([3, 1])
        
        with del_col1:
            delete_options = [f"{r['股票代码']} (Strike: {r['行权价 (Strike)']}, 到期: {r['到期日']})" for r in results]
            selected_option = st.selectbox("选择一笔已结清的持仓以供移除：", delete_options, label_visibility="collapsed")
            
        with del_col2:
            if st.button("🗑️ 确认删除", type="primary", use_container_width=True):
                selected_idx = delete_options.index(selected_option)
                target_real_id = results[selected_idx]["原始索引"]
                
                st.session_state.portfolio_data = st.session_state.portfolio_data.drop(target_real_id).reset_index(drop=True)
                save_data(st.session_state.portfolio_data)
                st.success("持仓已移除！")
                st.rerun()

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
                        
