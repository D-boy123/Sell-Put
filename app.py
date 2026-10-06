import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 设置网页标题和布局
st.set_page_config(page_title='Short Put 波动率扫描器', layout='wide')
st.title('📊 Short Put 选股扫描器 (Z-Score 优化多期版)')

# --- 侧边栏参数设置 ---
st.sidebar.header('⚙️ 扫描参数设置')
min_yield = st.sidebar.number_input(
    '最低年化收益率 (%)', min_value=0.0, max_value=100.0, value=20.0, step=1.0
)
min_gap = st.sidebar.number_input(
    '最小价差保护 (%)', min_value=0.0, max_value=10.0, value=2.0, step=0.5
)

# 默认监控池
default_watchlist = (
    'ABBV, AAPL, MSFT, INTC, QCOM, TQQQ, SPCX, PDD, TSLA, JD'
)
watchlist_input = st.sidebar.text_area('股票监控池 (逗号分隔)', default_watchlist)
watchlist = [t.strip().upper() for t in watchlist_input.split(',') if t.strip()]


# --- 核心扫描逻辑 ---
def scan_short_puts(tickers, min_annual_yield, min_gap_pct):
  today = dt.date.today()
  all_results = []

  for ticker_symbol in tickers:
    try:
      stock = yf.Ticker(ticker_symbol)

      # 1. 获取当前股价
      todays_data = stock.history(period='1d')
      if todays_data.empty:
        continue
      current_price = todays_data['Close'].iloc[-1]

      # 2. 获取期权到期日
      exp_dates = stock.options
      if not exp_dates:
        continue

      # 提取前两个到期日
      target_exps = exp_dates[:2]

      for index, target_exp in enumerate(target_exps):
        exp_date_obj = dt.datetime.strptime(target_exp, '%Y-%m-%d').date()
        days = (exp_date_obj - today).days
        if days <= 0:
          days = 1
        time_to_expiry_years = days / 365.0

        # 3. 获取 Put 期权链
        opt = stock.option_chain(target_exp)
        puts = opt.puts
        if 'strike' not in puts.columns or 'bid' not in puts.columns:
          continue

        # 4. 筛选
        otm_puts = puts[
            (puts['strike'] < current_price) & (puts['bid'] > 0)
        ].copy()

        for _, row in otm_puts.iterrows():
          strike = row['strike']
          bid = row['bid']
          iv = row.get('impliedVolatility', 0)

          if pd.isna(iv) or iv <= 0:
            iv = 0.25

          # 计算指标
          single_yield = bid / strike
          annual_yield = single_yield * (365 / days) * 100
          price_diff = current_price - strike
          z_score = price_diff / (
              current_price * iv * np.sqrt(time_to_expiry_years)
          )
          price_gap_pct = price_diff / current_price

          # 红线判定
          if (
              annual_yield >= min_annual_yield
              and price_gap_pct >= (min_gap_pct / 100.0)
          ):
            final_score = annual_yield * z_score
          else:
            final_score = 0.0

          all_results.append({
              'Ticker': ticker_symbol,
              'Current_Price': round(current_price, 2),
              'Expiration': target_exp,
              'Period_Index': index,  # 0代表近期，1代表下期
              'Days': days,
              'Strike': strike,
              'Bid': bid,
              'IV_%': round(iv * 100, 1),
              'Annual_Yield_%': round(annual_yield, 2),
              'Z_Score': round(z_score, 2),
              'Final_Score': round(final_score, 2),
          })
    except Exception:
      pass

  return pd.DataFrame(all_results) if all_results else pd.DataFrame()


# --- 高亮染色函数 ---
def style_row(row, best_near_idx, best_next_idx):
  """根据索引为整行施加不同的背景颜色"""
  if row.name == best_near_idx:
    # 近期最佳：浅绿色背景，深绿色文字
    return [
        'background-color: #d4edda; color: #155724; font-weight: bold'
    ] * len(row)
  elif row.name == best_next_idx:
    # 下期最佳：浅蓝色背景，深蓝色文字
    return [
        'background-color: #cce5ff; color: #004085; font-weight: bold'
    ] * len(row)
  return [''] * len(row)


# --- 页面主触发按钮 ---
if st.sidebar.button('🚀 开始扫描市场', type='primary'):
  with st.spinner('正在实时获取期权链数据，请稍候...'):
    results_df = scan_short_puts(watchlist, min_yield, min_gap)

  if not results_df.empty:
    st.subheader('📈 扫描结果展示')
    st.caption('💡 说明：绿色整行代表【近期最佳】，蓝色整行代表【下期最佳】。')

    for ticker in watchlist:
      ticker_df = results_df[
          (results_df['Ticker'] == ticker) & (results_df['Final_Score'] > 0)
      ]

      st.markdown(f'### 🔍 {ticker}')

      if not ticker_df.empty:
        unique_exps = sorted(ticker_df['Expiration'].unique())
        final_picks = []

        # 1. 提取每期前3名
        for exp in unique_exps:
          exp_df = ticker_df[ticker_df['Expiration'] == exp]
          top_picks_exp = exp_df.sort_values(
              by='Final_Score', ascending=False
          ).head(3)
          if not top_picks_exp.empty:
            final_picks.append(top_picks_exp)

        # 合并并重置索引，以便准确定位行号
        combined_df = (
            pd.concat(final_picks)
            .sort_values(
                by=['Expiration', 'Final_Score'], ascending=[True, False]
            )
            .reset_index(drop=True)
        )

        # 2. 找到近期和下期的最高分全局行索引
        best_near_idx = -1
        best_next_idx = -1

        near_part = combined_df[combined_df['Period_Index'] == 0]
        if not near_part.empty:
          best_near_idx = near_part['Final_Score'].idxmax()

        next_part = combined_df[combined_df['Period_Index'] == 1]
        if not next_part.empty:
          best_next_idx = next_part['Final_Score'].idxmax()

        # 3. 清洗并格式化输出表格
        print_df = combined_df[[
            'Expiration',
            'Days',
            'Strike',
            'Current_Price',
            'IV_%',
            'Bid',
            'Annual_Yield_%',
            'Z_Score',
            'Final_Score',
        ]]

        # 4. 调用 Pandas Styler 渲染颜色，并完美对齐
        styled_table = print_df.style.apply(
            style_row,
            axis=1,
            best_near_idx=best_near_idx,
            best_next_idx=best_next_idx,
        ).format({
            'Current_Price': '{:.2f}',
            'Strike': '{:.2f}',
            'Bid': '{:.2f}',
            'Annual_Yield_%': '{:.2f}',
            'IV_%': '{:.1f}',
            'Z_Score': '{:.2f}',
            'Final_Score': '{:.2f}',
        })

        # 5. 在网页上渲染成响应式表格
        st.dataframe(styled_table, use_container_width=True)
      else:
        st.info('未找到满足条件（年化收益或保护价差）的合格合约。')

      st.markdown('---')
  else:
    st.error('监控池内所有股票均未扫描到有效数据，请检查网络或稍后再试。')
else:
  st.info('👈 请在左侧调整参数，然后点击 **“开始扫描市场”** 按钮。')
