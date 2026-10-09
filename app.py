import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import os
import json
from datetime import datetime
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Sell Put 策略管理系统", page_icon="📈")

# ================= ================= =================
# 1. 本地数据持久化保存机制
# ================= ================= =================
DB_FILE = "portfolio.json"

def load_data():
    """从本地 JSON 文件读取持仓数据"""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                if "到期日" not in df.columns:
                    df["到期日"] = datetime.today().strftime('%Y-%m-%d')
                if "持仓数量(张)" not in df.columns:
                    df["持仓数量(张)"] = 1
                return df
        except:
            pass
    return pd.DataFrame([
        {"股票代码": "AAPL", "下单行权价(Strike)": 220.0, "收入权利金(Credit)": 1.50, "持仓数量(张)": 2, "到期日": "2026-11-20"},
        {"股票代码": "TSLA", "下单行权价(Strike)": 240.0, "收入权利金(Credit)": 3.80, "持仓数量(张)": 1, "到期日": "2026-11-20"}
    ])

def save_data(df):
    """将持仓数据保存到本地 JSON 文件"""
    df_to_save = df.reset_index(drop=True)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(df_to_save.to_dict(orient="records"), f, ensure_ascii=False, indent=4)

if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = load_data()


# ================= ================= =================
# 2. 核心数学模型：期权行权概率精算
# ================= ================= =================
def estimate_itm_probability_v2(ticker_symbol, strike, expiration_str):
    try:
        today = datetime.today().date()
        exp_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        
        ticker = yf.Ticker(ticker_symbol)
        current_price = ticker.fast_info['last_price']
        if dte <= 0:
            return (1.0 if strike > current_price else 0.0), dte
            
        T = dte / 365.0
        r = 0.04
        
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


# ================= ================= =================
# 3. 数据层封装函数
# ================= ================= =================
def fetch_portfolio_metrics(df):
    results = []
    total_credit_usd = 0.0
    high_risk_count = 0
    
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
            
            prob_str = "无法获取IV"
            if prob is not None:
                prob_val = prob * 100
                prob_str = f"{prob_val:.1f}%"
                if prob_val > 50.0:
                    high_risk_count += 1
            
            dte_str = f"{dte} 天" if isinstance(dte, int) else str(dte)
            
            results.append({
                "原始索引": index, "股票代码": ticker_str, "持仓数量 (张)": qty, "行权价 (Strike)": strike,
                "单张权利金 (Credit)": credit, "估计总权利金": f"${position_total_credit:.2f}",
                "当前正股价 (Current)": round(current_price, 2), "距行权安全垫 (%)": f"{price_diff_pct:.2f}%",
                "到期日": exp_str, "Remaining天数 (DTE)": dte_str, "预计被行权概率": prob_str, "勾选删除": False  
            })
        except:
            results.append({
                "原始索引": index, "股票代码": ticker_str, "持仓数量 (张)": qty, "行权价 (Strike)": strike,
                "单张权利金 (Credit)": credit, "估计总权利金": f"${position_total_credit:.2f}",
                "当前正股价 (Current)": "获取失败", "距行权安全垫 (%)": "N/A",
                "到期日": exp_str, "Remaining天数 (DTE)": "N/A", "预计被行权概率": "N/A", "勾选删除": False
            })
            
    return results, total_credit_usd, high_risk_count


def screen_potential_tickers(tickers, min_price):
    screen_results = []
    for t_sym in tickers:
        try:
            t_obj = yf.Ticker(t_sym)
            fast = t_obj.fast_info
            price = fast['last_price']
            
            if price < min_price:
                continue
                
            info = t_obj.info
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
    return screen_results


# ================= ================= =================
# 4. 视图渲染层封装 (将独立视图单独抽出，解决分支对齐漏洞)
# ================= ================= =================
def render_portfolio_view():
    st.title("💼 当前持仓动态透视")
    st.write("勾选表格最左侧并点击下方按钮，可完成移除持仓操作。")
    st.markdown("---")

    df = st.session_state.portfolio_data

    if df.empty:
        st.info("目前没有任何持仓数据，请在左侧侧边栏输入并录入新持仓。")
        return

    with st.spinner("正在获取实时股价，并结合 DTE 严密精算行权概率..."):
        results, total_credit_usd, high_risk_count = fetch_portfolio_metrics(df)

    res_df = pd.DataFrame(results)
    
    # 顶部宏观卡片看板
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="💼 运行中的期权单", value=f"{len(df)} 笔")
    with metric_col2:
        st.metric(label="💰 累计锁定权利金", value=f"${total_credit_usd:.2f}")
    with metric_col3:
        st.metric(label="🚨 处于高风险仓位 (>50%行权率)", value=f"{high_risk_count} 笔")
        
    st.markdown("")
    cols = ['勾选删除', '股票代码', '持仓数量 (张)', '行权价 (Strike)', '单张权利金 (Credit)', '估计总权利金', '当前正股价 (Current)', '距行权安全垫 (%)', '到期日', 'Remaining天数 (DTE)', '预计被行权概率']
    display_df = res_df[cols]
    
    st.subheader("📊 实时持仓监控盘面")
    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        disabled=['股票代码', '持仓数量 (张)', '行权价 (Strike)', '单张权利金 (Credit)', '估计总权利金', '当前正股价 (Current)', '距行权安全垫 (%)', '到期日', 'Remaining天数 (DTE)', '预计被行权概率'],
        key="portfolio_editor_v5"
    )
    
    st.markdown("")
    if st.button("🗑️ 删除表格选中持仓", type="primary"):
        selected_indices = edited_df[edited_df["勾选删除"] == True].index.tolist()
        if selected_indices:
            real_indices_to_drop = [res_df.iloc[i]["原始索引"] for i in selected_indices]
            st.session_state.portfolio_data = st.session_state.portfolio_data.drop(real_indices_to_drop).reset_index(drop=True)
            save_data(st.session_state.portfolio_data)
            st.success(f"成功删除 {len(real_indices_to_drop)} 个持仓项！")
            st.rerun()
        else:
            st.warning("⚠️ 请先在表格第一列中【勾选】您想要删除的股票，然后再点击本删除按钮。")


def render_screening_view():
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
            with st.spinner("正在连线雅虎财经..."):
                screen_results = screen_potential_tickers(tickers, min_price)
                    
            if screen_results:
                st.subheader("📊 扫描筛选结果透视表")
                st.dataframe(pd.DataFrame(screen_results), use_container_width=True)
            else:
                st.info("没有满足您条件的股票。")


# ================= ================= =================
# 5. 统一入口与侧边栏渲染控制流
# ================= ================= =================
st.sidebar.header("⚙️ 控制面板")
menu = st.sidebar.selectbox("功能菜单", ["💼 当前持仓管理", "🔍 筛选合适股票"])

st.sidebar.markdown("---")
st.sidebar.subheader("➕ 在此添加新持仓")
with st.sidebar.form(key="add_position_form", clear_on_submit=True):
