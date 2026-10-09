import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import os
import json
from datetime import datetime, date
from zoneinfo import ZoneInfo
from scipy.stats import norm
import altair as alt

st.set_page_config(layout="wide", page_title="Sell Put 策略管理系统", page_icon="📈")

# ================= 0. 全局常量 / 时区 =================
DB_FILE = "portfolio.json"
RISK_FREE_RATE = 0.045                        # 固定无风险利率（内部使用，不显示）
US_TZ = ZoneInfo("America/New_York")          # 美股东部时区


def us_today() -> date:
    """返回美国东部时区的当前日期（用于 DTE 计算）"""
    return datetime.now(US_TZ).date()


DEFAULT_COLUMNS = [
    "代码", "行权价", "到期日", "持仓数量(张)",
    "权利金(每股)", "开仓日期", "状态", "平仓价", "备注",
]


# ================= 1. 本地数据持久化保存机制 =================
def load_data():
    """从本地 JSON 文件读取持仓数据"""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame(columns=DEFAULT_COLUMNS)

            if "持仓数量(张)" not in df.columns:
                df["持仓数量(张)"] = 1
            if "到期日" not in df.columns:
                df["到期日"] = us_today().strftime("%Y-%m-%d")
            if "开仓日期" not in df.columns:
                df["开仓日期"] = us_today().strftime("%Y-%m-%d")
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


@st.cache_data(ttl=900, show_spinner=False)
def get_expirations(ticker: str):
    """获取标的全部到期日列表"""
    try:
        t = yf.Ticker(ticker)
        return list(t.options) if t.options else []
    except Exception:
        return []


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
def analyze_positions(df: pd.DataFrame) -> pd.DataFrame:
    """对每个持仓计算：现价、期权现价、浮盈、年化、行权概率等"""
    today = us_today()
    r = RISK_FREE_RATE
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

        quote = None
        if exp_date is not None and dte >= 0:
            quote = get_option_quote(ticker, exp_str, K)

        iv = quote["iv"] if quote and pd.notna(quote.get("iv", np.nan)) else hv
        theo, delta, gamma, theta, vega, prob_itm = bs_put(S, K, T, r, iv)

        mkt_price = quote["mid"] if quote else np.nan
        opt_price = mkt_price if pd.notna(mkt_price) else theo

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
            if pd.notna(opt_price) and not np.isnan(dte) and dte > 0:
                ann = (premium - opt_price) / K * 365.0 / dte
            elif not np.isnan(dte) and dte == 0:
                ann = 0.0
            else:
                ann = np.nan

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
            "IV": iv,
            "浮动盈亏($)": pnl,
            "占用资金($)": K * 100 * qty,
            "年化收益率": ann,
            "开仓年化": static_ann,
            "行权概率": prob_itm,
            "距行权价%": ((S - K) / S) if pd.notna(S) and S else np.nan,
            "Delta": delta,
            "Theta($/日)": (theta if pd.notna(theta) else 0) * 100 * qty,
        })

    return pd.DataFrame(out)


def fmt_pct(x):
    return f"{x*100:.2f}%" if pd.notna(x) else "—"


def fmt_money(x):
    return f"${x:,.2f}" if pd.notna(x) else "—"


# ================= 5. 选股分析工具函数 =================
def pick_expiration(ticker: str, target_dte: int):
    """按目标天数选择最接近的可用到期日"""
    exps = get_expirations(ticker)
    if not exps:
        return None
    today = us_today()
    best, best_diff = None, float("inf")
    for e in exps:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            dte = (d - today).days
            if dte < 1:
                continue
            diff = abs(dte - target_dte)
            if diff < best_diff:
                best_diff, best = diff, e
        except Exception:
            continue
    return best


def scan_ticker(ticker: str, expiry: str, otm_pct: float):
    """对单个标的做 Sell Put 扫描：找出最接近目标 OTM 的 Put 并计算指标"""
    mkt = fetch_market(ticker)
    if not mkt:
        return None
    S = mkt["price"]
    hv = mkt["hv"]

    puts = fetch_option_chain(ticker, expiry)
    if puts is None or puts.empty:
        return None

    # 只在 OTM（行权价 <= 现价）范围内选择
    puts = puts[puts["strike"] <= S * 0.999].copy()
    if puts.empty:
        return None

    target_K = S * (1 - otm_pct)
    idx = (puts["strike"] - target_K).abs().idxmin()
    row = puts.loc[idx]

    K = float(row["strike"])
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
    else:
        mid = last if pd.notna(last) and last > 0 else np.nan

    iv = row.get("impliedVolatility", np.nan)
    if pd.isna(iv) or iv <= 0:
        iv = hv

    try:
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte = (exp_date - us_today()).days
    except Exception:
        return None

    T = max(dte, 0) / 365.0
    theo, delta, gamma, theta, vega, prob_itm = bs_put(S, K, T, RISK_FREE_RATE, iv)

    prem = mid if pd.notna(mid) else theo
    ann = (prem / K * 365.0 / dte) if dte > 0 and pd.notna(prem) else np.nan

    return {
        "代码": ticker,
        "现价": S,
        "HV30": hv,
        "IV": iv,
        "IV/HV": (iv / hv) if pd.notna(hv) and hv > 0 else np.nan,
        "到期日": expiry,
        "剩余天数": dte,
        "建议行权价": K,
        "OTM%": (S - K) / S if S else np.nan,
        "Delta": delta,
        "权利金": prem,
        "权利金/张($)": prem * 100 if pd.notna(prem) else np.nan,
        "占用资金/张($)": K * 100,
        "年化收益率": ann,
        "行权概率": prob_itm,
        "OI": row.get("openInterest", np.nan),
        "成交量": row.get("volume", np.nan),
    }


# ================= 6. 主界面 =================
st.title("📈 Sell Put 策略管理系统")

if "df" not in st.session_state:
    st.session_state.df = load_data()
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 操作")
    if st.button("🔄 刷新行情缓存", use_container_width=True):
        st.cache_data.clear()
        st.success("缓存已清空")
        st.rerun()

    st.divider()
    st.caption(f"持仓记录：{len(st.session_state.df)} 条")
    st.caption(f"美东日期：{us_today():%Y-%m-%d}")

df = st.session_state.df

tab_overview, tab_manage, tab_screen = st.tabs(
    ["📊 持仓总览", "✏️ 持仓管理", "🔍 选股分析"]
)

# ---------- 计算分析结果 ----------
analysis = analyze_positions(df) if not df.empty else pd.DataFrame()

open_mask = analysis["状态"] == "持仓中" if not analysis.empty else pd.Series(dtype=bool)
open_pos = analysis[open_mask] if not analysis.empty else pd.DataFrame()

# ================= Tab 1: 持仓总览 =================
with tab_overview:
    if analysis.empty:
        st.info("暂无持仓数据，请到「✏️ 持仓管理」标签页添加你的第一笔 Sell Put。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        total_premium = open_pos["权利金"].mul(100).mul(open_pos["张数"]).sum() if not open_pos.empty else 0
        total_pnl = open_pos["浮动盈亏($)"].sum() if not open_pos.empty else 0
        total_cash = open_pos["占用资金($)"].sum() if not open_pos.empty else 0

        c1.metric("持仓中合约", f"{len(open_pos)} 笔")
        c2.metric("已收权利金", fmt_money(total_premium))
        c3.metric("浮动盈亏", fmt_money(total_pnl),
                  delta=f"{(total_pnl/total_cash*100):.2f}%" if total_cash else None)
        c4.metric("占用保证金", fmt_money(total_cash))

        st.divider()

        DISPLAY_COLS = [
            "代码", "状态", "行权价", "到期日", "剩余天数", "张数",
            "权利金", "现价", "期权现价", "IV",
            "浮动盈亏($)", "占用资金($)",
            "年化收益率", "开仓年化", "行权概率", "距行权价%",
        ]
        show = analysis[DISPLAY_COLS].copy()
        show["年化收益率"] = show["年化收益率"].apply(fmt_pct)
        show["开仓年化"] = show["开仓年化"].apply(fmt_pct)
        show["行权概率"] = show["行权概率"].apply(fmt_pct)
        show["距行权价%"] = show["距行权价%"].apply(fmt_pct)
        show["IV"] = show["IV"].apply(fmt_pct)

        st.dataframe(
            show.style.format({
                "行权价": "{:.2f}",
                "权利金": "{:.2f}",
                "现价": "{:.2f}",
                "期权现价": "{:.2f}",
                "浮动盈亏($)": "{:,.0f}",
                "占用资金($)": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True, height=420,
        )

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
        new_exp = f3.date_input("到期日", value=us_today())
        new_qty = f4.number_input("张数", min_value=1, value=1, step=1)

        f5, f6, f7 = st.columns(3)
        new_prem = f5.number_input("权利金 / 每股 ($)", min_value=0.0, value=1.00, step=0.05)
        new_open = f6.date_input("开仓日期", value=us_today())
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

# ================= Tab 3: 选股分析 =================
with tab_screen:
    st.subheader("🔍 Sell Put 选股扫描")
    st.caption(
        "输入标的清单，系统自动选择最接近目标到期天数的期权，"
        "并找出最接近目标 OTM 幅度的 Put，计算年化收益率与风险指标。"
    )

    with st.expander("📝 扫描参数", expanded=True):
        col_a, col_b = st.columns([3, 2])
        with col_a:
            tickers_input = st.text_area(
                "标的列表（逗号或换行分隔）",
                value="AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, SPY, QQQ",
                height=110,
            )
        with col_b:
            target_dte = st.slider("目标到期天数 (DTE)", 7, 180, 30, 7)
            otm_pct_input = st.slider("目标 OTM 幅度 (%)", 1, 40, 7, 1) / 100.0

        run_scan = st.button("🚀 开始扫描", type="primary", use_container_width=True)

    if run_scan:
        tickers = [t.strip().upper() for t in tickers_input.replace("\n", ",").split(",") if t.strip()]
        if not tickers:
            st.error("请输入至少一个标的代码")
        else:
            results = []
            progress = st.progress(0.0, text="扫描中…")
            for i, tk in enumerate(tickers):
                exp = pick_expiration(tk, target_dte)
                if exp is None:
                    progress.progress((i + 1) / len(tickers), text=f"扫描 {tk}（无可用期权）")
                    continue
                r = scan_ticker(tk, exp, otm_pct_input)
                if r:
                    results.append(r)
                progress.progress((i + 1) / len(tickers), text=f"扫描 {tk} 完成")
            progress.empty()

            if results:
                res_df = pd.DataFrame(results).sort_values(
                    "年化收益率", ascending=False).reset_index(drop=True)
                st.session_state.scan_results = res_df
            else:
                st.session_state.scan_results = None
                st.warning("未获取到任何数据，请检查标的代码或稍后重试。")

    if st.session_state.scan_results is not None:
        res = st.session_state.scan_results.copy()

        st.divider()
        f1, f2, f3 = st.columns(3)
        min_ann = f1.slider("最低年化收益率 (%)", 0, 100, 15, 1) / 100.0
        min_ivhv = f2.slider("最低 IV/HV 比率", 0.0, 3.0, 0.0, 0.1)
        max_delta = f3.slider("最大 |Delta|", 0.05, 0.60, 0.40, 0.01)

        filt = res[
            (res["年化收益率"].fillna(-1) >= min_ann)
            & (res["IV/HV"].fillna(0) >= min_ivhv)
            & (res["Delta"].abs().fillna(1) <= max_delta)
        ].reset_index(drop=True)

        st.caption(f"共 {len(filt)} / {len(res)} 个标的符合筛选条件")

        if filt.empty:
            st.info("没有符合条件的标的，请放宽筛选条件。")
        else:
            # ---- 图表：年化收益率排序 ----
            chart = alt.Chart(filt).mark_bar(color="#2E86DE").encode(
                x=alt.X("代码:N", sort="-y", title="标的"),
                y=alt.Y("年化收益率:Q", title="年化收益率"),
                tooltip=["代码", alt.Tooltip("年化收益率:Q", format=".2%"),
                         alt.Tooltip("IV/HV:Q", format=".2f"),
                         alt.Tooltip("Delta:Q", format=".2f")],
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

            # ---- 结果表 ----
            show = filt.copy()
            show["年化收益率"] = show["年化收益率"].apply(fmt_pct)
            show["行权概率"] = show["行权概率"].apply(fmt_pct)
            show["OTM%"] = show["OTM%"].apply(fmt_pct)
            show["HV30"] = show["HV30"].apply(fmt_pct)
            show["IV"] = show["IV"].apply(fmt_pct)

            st.dataframe(
                show.style.format({
                    "现价": "{:.2f}",
                    "IV/HV": "{:.2f}",
                    "建议行权价": "{:.2f}",
                    "Delta": "{:.3f}",
                    "权利金": "{:.2f}",
                    "权利金/张($)": "{:,.0f}",
                    "占用资金/张($)": "{:,.0f}",
                    "OI": "{:,.0f}",
                    "成交量": "{:,.0f}",
                }, na_rep="—"),
                use_container_width=True, height=420,
            )

            st.caption(
                "💡 IV/HV > 1 表示期权隐含波动率高于实际历史波动率，卖出 Put 相对更有优势；"
                "|Delta| 越小则被行权概率越低。行情可能存在延迟，请结合券商实际报价下单。"
            )
