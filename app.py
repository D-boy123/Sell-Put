import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import os
import json
from datetime import datetime, date
from scipy.stats import norm
import altair as alt

st.set_page_config(layout="wide", page_title="Sell Put 策略管理系统", page_icon="📈")

# ================= 1. 本地数据持久化保存机制 =================
DB_FILE = "portfolio.json"

DEFAULT_COLUMNS = [
    "代码", "行权价", "到期日", "持仓数量(张)",
    "权利金(每股)", "开仓日期", "状态", "平仓价", "备注",
]


def load_data():
    """从本地 JSON 文件读取持仓数据"""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame(columns=DEFAULT_COLUMNS)

            # 兼容升级旧数据机制
            if "持仓数量(张)" not in df.columns:
                df["持仓数量(张)"] = 1
            if "到期日" not in df.columns:
                df["到期日"] = datetime.today().strftime("%Y-%m-%d")
            if "开仓日期" not in df.columns:
                df["开仓日期"] = datetime.today().strftime("%Y-%m-%d")
            if "状态" not in df.columns:
                df["状态"] = "持仓中"
            if "平仓价" not in df.columns:
                df["平仓价"] = np.nan
            if "备注" not in df.columns:
                df["备注"] = ""

            for col in DEFAULT_COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            return df[DEFAULT_COLUMNS]
        except Exception as e:
            st.error(f"读取本地数据失败：{e}")
    return pd.DataFrame(columns=DEFAULT_COLUMNS)


def save_data(df: pd.DataFrame):
    """写回本地 JSON 文件（自动处理 NaN / numpy 类型）"""
    records = json.loads(df.to_json(orient="records", force_ascii=False))
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ================= 2. 行情数据获取（带缓存） =================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_market(ticker: str):
    """获取标的最新价 + 30日历史波动率"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y", auto_adjust=True)
        if hist.empty or len(hist) < 5:
            return None
        close = hist["Close"].dropna()
        S = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else S
        ret = np.log(close / close.shift(1)).dropna()
        window = min(30, len(ret))
        hv = float(ret.tail(window).std() * np.sqrt(252)) if window > 2 else 0.30
        return {"price": S, "prev_close": prev, "hv": hv}
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_option_chain(ticker: str, expiry: str):
    """获取指定到期日的 Put 期权链"""
    try:
        t = yf.Ticker(ticker)
        return t.option_chain(expiry).puts
    except Exception:
        return pd.DataFrame()


def get_option_quote(ticker: str, expiry: str, strike: float):
    """取最接近行权价的 Put 报价（优先买卖中间价）"""
    puts = fetch_option_chain(ticker, expiry)
    if puts is None or puts.empty:
        return None
    try:
        idx = (puts["strike"] - float(strike)).abs().idxmin()
        row = puts.loc[idx]
    except Exception:
        return None

    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)
    if pd.notna(bid) and pd.notna(ask) and ask > 0 and bid > 0:
        mid = (bid + ask) / 2.0
    else:
        mid = last if pd.notna(last) and last > 0 else np.nan

    iv = row.get("impliedVolatility", np.nan)
    return {
        "mid": float(mid) if pd.notna(mid) else np.nan,
        "iv": float(iv) if pd.notna(iv) and iv > 0 else np.nan,
        "oi": row.get("openInterest", np.nan),
        "volume": row.get("volume", np.nan),
    }


# ================= 3. Black-Scholes 定价与希腊字母 =================
def bs_put(S, K, T, r, sigma):
    """返回 (理论价, delta, gamma, theta每日, vega每1%, 行权概率)"""
    if S is None or np.isnan(S):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if T is None or np.isnan(T) or T <= 0:
        intrinsic = max(K - S, 0.0)
        delta = -1.0 if S < K else 0.0
        prob = 1.0 if S < K else 0.0
        return intrinsic, delta, 0.0, 0.0, 0.0, prob

    if sigma is None or np.isnan(sigma) or sigma <= 0:
        sigma = 0.30

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    prob_itm = norm.cdf(-d2)
    return price, delta, gamma, theta, vega, prob_itm


# ================= 4. 持仓分析核心 =================
def analyze_positions(df: pd.DataFrame, r: float) -> pd.DataFrame:
    """对每个持仓计算：现价、期权现价、浮盈、年化、希腊字母等"""
    today = date.today()
    out = []

    for _, pos in df.iterrows():
        ticker = str(pos["代码"]).strip().upper()
        try:
            K = float(pos["行权价"])
            qty = int(pos["持仓数量(张)"])
            premium = float(pos["权利金(每股)"])
        except Exception:
            continue

        exp_str = str(pos["到期日"])[:10]
        status = str(pos.get("状态", "持仓中"))

        mkt = fetch_market(ticker)
        S = mkt["price"] if mkt else np.nan
        hv = mkt["hv"] if mkt else np.nan

        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
        except Exception:
            exp_date, dte = None, np.nan

        T = max(dte, 0) / 365.0 if not np.isnan(dte) else np.nan

        # 期权现价：优先市场报价，否则用 BS 理论价
        quote = None
        if exp_date is not None and dte >= 0:
            quote = get_option_quote(ticker, exp_str, K)

        iv = quote["iv"] if quote and pd.notna(quote.get("iv", np.nan)) else hv
        theo, delta, gamma, theta, vega, prob_itm = bs_put(S, K, T, r, iv)

        mkt_price = quote["mid"] if quote else np.nan
        opt_price = mkt_price if pd.notna(mkt_price) else theo

        # 盈亏
        if status == "已平仓":
            close_px = pos.get("平仓价", np.nan)
            try:
                close_px = float(close_px)
            except Exception:
                close_px = np.nan
            pnl = (premium - close_px) * 100 * qty if pd.notna(close_px) else np.nan
            ann = np.nan
        else:
            pnl = (premium - opt_price) * 100 * qty if pd.notna(opt_price) else np.nan
            # 以行权价占用的现金为分母的年化收益率（剩余期间）
            if pd.notna(opt_price) and not np.isnan(dte) and dte > 0:
                ann = (premium - opt_price) / K * 365.0 / dte
            elif not np.isnan(dte) and dte == 0:
                ann = 0.0
            else:
                ann = np.nan

        # 开仓时静态年化（基于权利金 / 行权价）
        try:
            open_date = datetime.strptime(str(pos["开仓日期"])[:10], "%Y-%m-%d").date()
            dte_open = (exp_date - open_date).days if exp_date else np.nan
        except Exception:
            dte_open = np.nan
        static_ann = (premium / K * 365.0 / dte_open) if dte_open and dte_open > 0 else np.nan

        out.append({
            "代码": ticker,
            "状态": status,
            "行权价": K,
            "到期日": exp_str,
            "剩余天数": dte,
            "张数": qty,
            "权利金": premium,
            "现价": S,
            "期权现价": opt_price,
            "市场价": mkt_price,
            "IV": iv,
            "浮动盈亏($)": pnl,
            "最大收益($)": premium * 100 * qty,
            "占用资金($)": K * 100 * qty,
            "年化收益率": ann,
            "开仓年化": static_ann,
            "Delta": delta,
            "Gamma": gamma,
            "Theta($/日)": (theta if pd.notna(theta) else 0) * 100 * qty,
            "Vega($/1%)": (vega if pd.notna(vega) else 0) * 100 * qty,
            "行权概率": prob_itm,
            "距行权价%": ((S - K) / S) if pd.notna(S) and S else np.nan,
            "盈亏平衡点": K - premium,
            "Delta敞口($)": (delta * 100 * qty * S) if pd.notna(delta) and pd.notna(S) else np.nan,
        })

    return pd.DataFrame(out)


def fmt_pct(x):
    return f"{x*100:.2f}%" if pd.notna(x) else "—"


def fmt_money(x):
    return f"${x:,.2f}" if pd.notna(x) else "—"


# ================= 5. 主界面 =================
st.title("📈 Sell Put 策略管理系统")

if "df" not in st.session_state:
    st.session_state.df = load_data()

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 全局设置")
    rf_pct = st.number_input("无风险利率 (%)", value=4.50, step=0.05, format="%.2f")
    r = rf_pct / 100.0

    st.divider()
    if st.button("🔄 刷新行情缓存", use_container_width=True):
        st.cache_data.clear()
        st.success("缓存已清空")
        st.rerun()

    st.divider()
    st.caption(f"数据文件：`{os.path.abspath(DB_FILE)}`")
    st.caption(f"持仓记录：{len(st.session_state.df)} 条")

df = st.session_state.df

tab_overview, tab_manage, tab_risk, tab_data = st.tabs(
    ["📊 持仓总览", "✏️ 持仓管理", "⚠️ 风险分析", "💾 数据管理"]
)

# ---------- 计算分析结果 ----------
analysis = analyze_positions(df, r) if not df.empty else pd.DataFrame()

open_mask = analysis["状态"] == "持仓中" if not analysis.empty else pd.Series(dtype=bool)
open_pos = analysis[open_mask] if not analysis.empty else pd.DataFrame()

# ================= Tab 1: 持仓总览 =================
with tab_overview:
    if analysis.empty:
        st.info("暂无持仓数据，请到「✏️ 持仓管理」标签页添加你的第一笔 Sell Put。")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        total_premium = open_pos["权利金"].mul(100).mul(open_pos["张数"]).sum() if not open_pos.empty else 0
        total_pnl = open_pos["浮动盈亏($)"].sum() if not open_pos.empty else 0
        total_cash = open_pos["占用资金($)"].sum() if not open_pos.empty else 0
        total_theta = open_pos["Theta($/日)"].sum() if not open_pos.empty else 0
        avg_ann = open_pos["年化收益率"].mean() if not open_pos.empty else np.nan

        c1.metric("持仓中合约", f"{len(open_pos)} 笔")
        c2.metric("已收权利金", fmt_money(total_premium))
        c3.metric("浮动盈亏", fmt_money(total_pnl),
                  delta=f"{(total_pnl/total_cash*100):.2f}%" if total_cash else None)
        c4.metric("占用保证金", fmt_money(total_cash))
        c5.metric("组合 Theta", f"${total_theta:,.2f}/日")

        st.divider()

        show = analysis.copy()
        show["年化收益率"] = show["年化收益率"].apply(fmt_pct)
        show["开仓年化"] = show["开仓年化"].apply(fmt_pct)
        show["行权概率"] = show["行权概率"].apply(fmt_pct)
        show["距行权价%"] = show["距行权价%"].apply(fmt_pct)
        show["IV"] = show["IV"].apply(fmt_pct)

        st.dataframe(
            show.style.format({
                "行权价": "{:.2f}", "权利金": "{:.2f}", "现价": "{:.2f}",
                "期权现价": "{:.2f}", "市场价": "{:.2f}",
                "浮动盈亏($)": "{:,.0f}", "最大收益($)": "{:,.0f}",
                "占用资金($)": "{:,.0f}", "Theta($/日)": "{:,.1f}",
                "Vega($/1%)": "{:,.1f}", "Delta": "{:.3f}", "Gamma": "{:.4f}",
                "盈亏平衡点": "{:.2f}", "Delta敞口($)": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True, height=420,
        )

        # ---- 单笔到期损益图 ----
        st.subheader("📉 到期损益模拟")
        if not open_pos.empty:
            labels = open_pos.apply(
                lambda x: f"{x['代码']} {x['行权价']:.1f}P {x['到期日']}", axis=1).tolist()
            sel = st.selectbox("选择持仓", labels, key="payoff_sel")
            row = open_pos.iloc[labels.index(sel)]

            S0 = row["现价"] if pd.notna(row["现价"]) else row["行权价"]
            K, prem, qty = row["行权价"], row["权利金"], row["张数"]
            lo, hi = S0 * 0.6, S0 * 1.4
            grid = np.linspace(lo, hi, 200)
            payoff = np.where(grid >= K, prem, prem - (K - grid)) * 100 * qty

            chart_df = pd.DataFrame({"股价": grid, "损益": payoff})
            base = alt.Chart(chart_df).mark_line(
                color="#2E86DE", strokeWidth=2
            ).encode(
                x=alt.X("股价:Q", title="到期股价 ($)"),
                y=alt.Y("损益:Q", title="损益 ($)"),
            )
            zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                strokeDash=[4, 4], color="gray").encode(y="y:Q")
            strike_rule = alt.Chart(pd.DataFrame({"x": [K]})).mark_rule(
                strokeDash=[6, 4], color="red").encode(x="x:Q")
            be_rule = alt.Chart(pd.DataFrame({"x": [K - prem]})).mark_rule(
                strokeDash=[6, 4], color="green").encode(x="x:Q")
            spot_rule = alt.Chart(pd.DataFrame({"x": [S0]})).mark_rule(
                strokeDash=[2, 4], color="black").encode(x="x:Q")

            st.altair_chart(
                (base + zero_rule + strike_rule + be_rule + spot_rule).properties(height=420),
                use_container_width=True,
            )

# ================= Tab 2: 持仓管理 =================
with tab_manage:
    st.subheader("➕ 新建 Sell Put 持仓")
    with st.form("add_position", clear_on_submit=True):
        f1, f2, f3, f4 = st.columns(4)
        new_ticker = f1.text_input("标的代码", placeholder="例如 AAPL / NVDA / SPY").upper().strip()
        new_strike = f2.number_input("行权价 ($)", min_value=0.01, value=100.0, step=1.0)
        new_exp = f3.date_input("到期日", value=date.today())
        new_qty = f4.number_input("张数", min_value=1, value=1, step=1)

        f5, f6, f7 = st.columns(3)
        new_prem = f5.number_input("权利金 / 每股 ($)", min_value=0.0, value=1.00, step=0.05)
        new_open = f6.date_input("开仓日期", value=date.today())
        new_note = f7.text_input("备注", placeholder="可选")

        submitted = st.form_submit_button("✅ 添加持仓", use_container_width=True)

        if submitted:
            if not new_ticker:
                st.error("请填写标的代码")
            else:
                new_row = {
                    "代码": new_ticker,
                    "行权价": float(new_strike),
                    "到期日": new_exp.strftime("%Y-%m-%d"),
                    "持仓数量(张)": int(new_qty),
                    "权利金(每股)": float(new_prem),
                    "开仓日期": new_open.strftime("%Y-%m-%d"),
                    "状态": "持仓中",
                    "平仓价": np.nan,
                    "备注": new_note,
                }
                st.session_state.df = pd.concat(
                    [st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(st.session_state.df)
                st.success(f"已添加 {new_ticker} {new_strike}P")
                st.rerun()

    st.divider()
    st.subheader("📝 编辑 / 删除持仓")
    if df.empty:
        st.info("暂无持仓可编辑。")
    else:
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            height=380,
            column_config={
                "状态": st.column_config.SelectboxColumn(
                    "状态", options=["持仓中", "已平仓"], required=True),
                "行权价": st.column_config.NumberColumn("行权价", format="%.2f"),
                "权利金(每股)": st.column_config.NumberColumn("权利金(每股)", format="%.2f"),
                "平仓价": st.column_config.NumberColumn("平仓价", format="%.2f"),
                "持仓数量(张)": st.column_config.NumberColumn("持仓数量(张)", min_value=1, step=1),
            },
            key="editor",
        )
        b1, b2 = st.columns([1, 4])
        if b1.button("💾 保存修改", type="primary", use_container_width=True):
            st.session_state.df = edited.reset_index(drop=True)
            save_data(st.session_state.df)
            st.cache_data.clear()
            st.success("已保存")
            st.rerun()
        if b2.button("↩️ 放弃修改", use_container_width=False):
            st.rerun()

# ================= Tab 3: 风险分析 =================
with tab_risk:
    if open_pos.empty:
        st.info("暂无持仓中的合约。")
    else:
        st.subheader("组合风险指标")
        total_cash = open_pos["占用资金($)"].sum()
        total_delta = open_pos["Delta敞口($)"].sum()
        weighted_prob = (
            (open_pos["行权概率"].fillna(0) * open_pos["占用资金($)"]).sum() / total_cash
            if total_cash else np.nan
        )
        near = open_pos[open_pos["距行权价%"] < 0.05]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("总占用资金", fmt_money(total_cash))
        k2.metric("组合 Delta 敞口", fmt_money(total_delta))
        k3.metric("加权行权概率", fmt_pct(weighted_prob))
        k4.metric("接近行权(<5%)", f"{len(near)} 笔")

        st.divider()
        st.subheader("🚨 风险预警")
        warnings = []
        for _, row in open_pos.iterrows():
            tag = f"**{row['代码']} {row['行权价']:.1f}P {row['到期日']}**"
            if pd.notna(row["距行权价%"]) and row["距行权价%"] < 0.03:
                warnings.append(f"🔴 {tag}：现价距行权价仅 {row['距行权价%']*100:.2f}%，深度实值风险")
            elif pd.notna(row["距行权价%"]) and row["距行权价%"] < 0.08:
                warnings.append(f"🟠 {tag}：现价距行权价 {row['距行权价%']*100:.2f}%，需密切关注")
            if pd.notna(row["剩余天数"]) and 0 <= row["剩余天数"] <= 7:
                warnings.append(f"⏰ {tag}：仅剩 {int(row['剩余天数'])} 天到期，考虑平仓或滚动")
            if pd.notna(row["剩余天数"]) and row["剩余天数"] < 0:
                warnings.append(f"⚫ {tag}：已过期 {abs(int(row['剩余天数']))} 天，请更新状态")

        if warnings:
            for w in warnings:
                st.warning(w, icon="⚠️")
        else:
            st.success("当前所有持仓风险指标正常 ✅")

        st.divider()
        st.subheader("按标的聚合")
        agg = open_pos.groupby("代码").agg(
            张数=("张数", "sum"),
            占用资金=("占用资金($)", "sum"),
            浮动盈亏=("浮动盈亏($)", "sum"),
            总Delta=("Delta敞口($)", "sum"),
            总Theta=("Theta($/日)", "sum"),
        ).reset_index()
        st.dataframe(
            agg.style.format({
                "占用资金": "${:,.0f}", "浮动盈亏": "${:,.0f}",
                "总Delta": "${:,.0f}", "总Theta": "${:,.1f}",
            }),
            use_container_width=True,
        )

# ================= Tab 4: 数据管理 =================
with tab_data:
    st.subheader("💾 数据导入 / 导出")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "⬇️ 导出 JSON",
            data=json.dumps(json.loads(df.to_json(orient="records", force_ascii=False)),
                            ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"portfolio_{datetime.today():%Y%m%d}.json",
            mime="application/json",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "⬇️ 导出 CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"portfolio_{datetime.today():%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        if st.button("🗑️ 清空所有数据", use_container_width=True):
            st.session_state.df = pd.DataFrame(columns=DEFAULT_COLUMNS)
            save_data(st.session_state.df)
            st.warning("已清空全部持仓")
            st.rerun()

    st.divider()
    st.subheader("📤 导入持仓文件")
    uploaded = st.file_uploader("选择 JSON 或 CSV 文件", type=["json", "csv"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".json"):
                incoming = pd.DataFrame(json.load(uploaded))
            else:
                incoming = pd.read_csv(uploaded)
            for col in DEFAULT_COLUMNS:
                if col not in incoming.columns:
                    incoming[col] = np.nan
            mode = st.radio("导入方式", ["覆盖现有数据", "追加到现有数据"], horizontal=True)
            if st.button("确认导入", type="primary"):
                if mode == "覆盖现有数据":
                    st.session_state.df = incoming[DEFAULT_COLUMNS].reset_index(drop=True)
                else:
                    st.session_state.df = pd.concat(
                        [st.session_state.df, incoming[DEFAULT_COLUMNS]], ignore_index=True)
                save_data(st.session_state.df)
                st.success(f"成功导入 {len(incoming)} 条记录")
                st.rerun()
        except Exception as e:
            st.error(f"文件解析失败：{e}")

    st.divider()
    st.caption(
        "💡 提示：所有数据保存在本地 `portfolio.json` 文件中，不会上传到任何服务器。"
        "行情与期权数据来自 Yahoo Finance，可能存在 15 分钟延迟。"
    )
