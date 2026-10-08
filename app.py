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
        
        opt = ticker.option_chain(expirations[0]) # 取最近一个到期日进行估算
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

# ================= 3. 侧边栏功能切换菜单 =================
menu = st.sidebar.selectbox("功能菜单", ["🔍 筛选合适股票", "💼 当前持仓管理", "📈 策略主页简介"])

# ================= 模块一：🔍 筛选合适股票（完全恢复并保留） =================
if menu == "🔍 筛选合适股票":
    st.title("🔍 Sell Put 潜在股票筛选神器")
    st.write("输入您感兴趣的股票代码，系统将帮您抓取核心行情指标，辅助评估其是否适合作为 Sell Put 标的。")
    
    # 股票池输入与核心参数设定
    col_input, col_param = st.columns([2, 1])
    with col_input:
        ticker_input = st.text_input("请输入股票代码（多个请用逗号隔开，例如: AAPL, TSLA, NVDA, AMD）", "AAPL, TSLA, NVDA")
    with col_param:
        min_price = st.number_input("最低股价过滤 ($)", min_value=0.0, value=50.0)

    # 触发筛选按钮
    if st.button("🚀 开始抓取并筛选数据", type="primary"):
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        
        if not tickers:
            st.error("请输入至少一个股票代码！")
        else:
            screen_results = []
            with st.spinner("正在连线雅虎财经，全面扫描正股指标中..."):
                for t_sym in tickers:
                    try:
                        t_obj = yf.Ticker(t_sym)
                        info = t_obj.info
                        fast = t_obj.fast_info
                        
                        price = fast['last_price']
                        
                        # 过滤低于设定阈值的股票
                        if price < min_price:
                            continue
                            
                        # 安全提取雅虎财经财务/技术面指标
                        pe = info.get('trailingPE', np.nan)
                        pe_str = f"{pe:.1f}" if pd.notnull(pe) else "N/A"
                        
                        fifty_two_week_low = info.get('fiftyTwoWeekLow', np.nan)
                        dist_from_low = ((price - fifty_two_week_low) / fifty_two_week_low * 100) if pd.notnull(fifty_two_week_low) else np.nan
                        dist_str = f"+{dist_from_low:.1f}%" if pd.notnull(dist_from_low) else "N/A"
                        
                        beta = info.get('beta', np.nan)
                        beta_str = f"{beta:.2f}" if pd.notnull(beta) else "N/A"
                        
                        # 简单评估建议（Sell Put 偏好：市盈率健康、波动稳定、距离52周低点有一定安全垫）
                        if pd.notnull(beta) and beta > 1.5:
                            advice = "⚠️ 波动剧烈 (高Beta)，权利金高但接盘风险大"
                        elif pd.notnull(pe) and pe > 50:
                            advice = "⚠️ 估值偏高 (高PE)，注意高位回撤风险"
                        else:
                            advice = "✅ 适合 Sell Put (估值或波动较稳健)"

                        screen_results.append({
                            "股票代码": t_sym,
                            "当前股价": f"${price:.2f}",
                            "市盈率 (PE)": pe_str,
                            "52周最低价距离": dist_str,
                            "波动率系数 (Beta)": beta_str,
                            "策略初评建议": advice
                        })
                    except:
                        screen_results.append({
                            "股票代码": t_sym, "当前股价": "抓取失败", "市盈率 (PE)": "N/A", "52周最低价距离": "N/A", "波动率系数 (Beta)": "N/A", "策略初评建议": "❌ 无法获取该股票信息"
                        })
                        
            if screen_results:
                st.subheader("📊 扫描筛选结果透视表")
                st.dataframe(pd.DataFrame(screen_results), use_container_width=True)
            else:
                st.info("没有满足您所设定『最低股价过滤』条件的股票。")

# ================= 模块二：💼 当前持仓管理（今日最新弹窗+永久保存版） =================
elif menu == "💼 当前持仓管理":
    st.title("💼 当前持仓动态透视")
    st.write("您可以在这里查看实时仓位表现、计算安全垫和行权率，并使用底部的弹窗添加或一键删除单子。")
    
    # --- 操作按钮区域：添加新持仓 ---
    if st.button("➕ 添加新持仓（弹窗输入）"):
        st.dialog("add_position_modal") 
        
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
        with st.spinner("正在获取实时股价，并利用 Black-Scholes 模型严密推算行权概率..."):
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
                        "ID": index,
                        "股票代码": ticker_str,
                        "行权价 (Strike)": strike,
                        "权利金 (Credit)": credit,
                        "当前估价 (Current)": round(current_price, 2),
                        "距行权安全垫 (%)": f"{price_diff_pct:.2f}%",
                        "预计被行权概率": prob_str
                    })
                except:
                    results.append({
                        "ID": index, "股票代码": ticker_str, "行权价 (Strike)": strike, "权利金 (Credit)": credit, "当前估价 (Current)": "获取失败", "距行权安全垫 (%)": "N/A", "预计被行权概率": "N/A"
                    })

        res_df = pd.DataFrame(results)
        display_df = res_df.drop(columns=["ID"]) 
        st.subheader("📊 实时持仓监控盘面")
        st.dataframe(display_df, use_container_width=True)

        # --- 移除单子区域 ---
        st.markdown("---")
        st.subheader("🗑️ 移除已结清仓位")
        delete_list = [f"{r['股票代码']} (Strike: {r['行权价 (Strike)']})" for r in results]
        selected_to_delete = st.selectbox("选择需要删除的持仓项：", delete_list)
        
        if st.button("🔥 确认删除选中持仓", type="primary"):
            selected_index = delete_list.index(selected_to_delete)
            target_id = results[selected_index]["ID"]
            
            st.session_state.portfolio_data = st.session_state.portfolio_data.drop(target_id).reset_index(drop=True)
            save_data(st.session_state.portfolio_data)
            st.success("持仓已删除，本地存档已同步更新！")
            st.rerun()

# ================= 模块三：📈 策略主页简介 =================
elif menu == "📈 策略主页简介":
    st.title("📈 Sell Put (Short Put) 期权策略看板")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📊 跟踪的持仓标的", value=f"{len(st.session_state.portfolio_data)} 个")
    with col2:
        total_credit = st.session_state.portfolio_data["收入权利金(Credit)"].sum() if not st.session_state.portfolio_data.empty else 0
        st.metric(label="💰 已落袋/锁定权利金总额", value=f"${total_credit:.2f}")

    st.markdown("---")
    st.subheader("💡 策略通俗释义")
    st.write("卖出看跌期权（Sell Put）核心逻辑是：**『承诺在未来某个低价向别人买入股票，并当场收取一笔保管费（权利金）。』**")
    st.markdown(
        """
        * **完美结局（股价横盘或上涨）**：期权归零，保管费纯赚，无需买入股票。
        * **抄底结局（股价小幅跌破行权价）**：被迫以你心仪的低价买入股票，且扣除保管费后，实际接盘成本更低。
        * **爆雷危机（股价雪崩）**：股价跌幅远超想象，必须高价接盘，会产生账面浮亏。
        """
    )
