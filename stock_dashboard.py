import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

import pipeline

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


class StockAnalyzer:
    """Thin orchestration over the fetch / indicator / ML pipeline."""

    def __init__(self, fetcher=None):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.fetcher = fetcher or pipeline.default_fetcher

    def fetch_stock_data(self, symbol, start, end, adjust="qfq",
                         force_refresh=False):
        """Fetch through the (ticker, start, end, adjust) isolated cache."""
        return self.fetcher.fetch(
            symbol, start, end, adjust=adjust, force_refresh=force_refresh
        )

    def calculate_technical_indicators(self, data):
        return pipeline.calculate_technical_indicators(data)

    def prepare_ml_features(self, data):
        return pipeline.prepare_ml_features(data)

    def train_prediction_model(self, data):
        """Train on the leak-free aligned features / forward label."""
        x, y = pipeline.build_prediction_dataset(data)

        if len(x) < 100:
            return None

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        self.model.fit(x_train_scaled, y_train)

        return {
            "train_score": self.model.score(x_train_scaled, y_train),
            "test_score": self.model.score(x_test_scaled, y_test),
            "feature_importance": dict(
                zip(x.columns, self.model.feature_importances_)
            ),
            "last_features": x.iloc[-1:],
            "feature_cols": list(x.columns),
            "x_test": x_test.sort_index(),
            "y_test": y_test.sort_index(),
        }

    def predict_next_price(self, model_info):
        if model_info is None:
            return None
        scaled = self.scaler.transform(model_info["last_features"])
        return self.model.predict(scaled)[0]

    def generate_market_analysis(self, data, info, symbol):
        """Generate AI-powered market analysis."""
        latest = data.iloc[-1]
        prev = data.iloc[-2]

        price_change = latest["Close"] - prev["Close"]
        price_change_pct = price_change / prev["Close"] * 100

        rsi = latest.get("RSI", 50)
        rsi = 50 if pd.isna(rsi) else rsi
        sma_20 = latest.get("SMA_20", latest["Close"])
        sma_20 = latest["Close"] if pd.isna(sma_20) else sma_20
        sma_50 = latest.get("SMA_50", latest["Close"])
        sma_50 = latest["Close"] if pd.isna(sma_50) else sma_50
        bb_upper = latest.get("BB_upper", latest["Close"])
        bb_lower = latest.get("BB_lower", latest["Close"])

        avg_volume = data["Volume"].rolling(20).mean().iloc[-1]
        if avg_volume and avg_volume > 0:
            volume_ratio = latest["Volume"] / avg_volume
        else:
            volume_ratio = 1

        macd = latest.get("MACD", 0)
        macd = 0 if pd.isna(macd) else macd
        macd_signal = latest.get("MACD_signal", 0)
        macd_signal = 0 if pd.isna(macd_signal) else macd_signal

        analysis = []

        if price_change_pct > 3:
            analysis.append(f"🚀 {symbol} shows exceptional bullish momentum with a {price_change_pct:.2f}% surge")
        elif price_change_pct > 1:
            analysis.append(f"🟢 {symbol} demonstrates strong upward movement (+{price_change_pct:.2f}%)")
        elif price_change_pct > 0:
            analysis.append(f"🟡 {symbol} shows modest gains (+{price_change_pct:.2f}%)")
        elif price_change_pct > -1:
            analysis.append(f"🟡 {symbol} experiences slight decline ({price_change_pct:.2f}%)")
        elif price_change_pct > -3:
            analysis.append(f"🔴 {symbol} shows moderate bearish pressure ({price_change_pct:.2f}%)")
        else:
            analysis.append(f"🔻 {symbol} faces significant selling pressure ({price_change_pct:.2f}%)")

        if rsi > 80:
            analysis.append(f"🚨 RSI at {rsi:.1f} indicates severely overbought conditions - potential reversal ahead")
        elif rsi > 70:
            analysis.append(f"⚠️ RSI at {rsi:.1f} shows overbought territory - exercise caution")
        elif rsi < 20:
            analysis.append(f"🛒 RSI at {rsi:.1f} signals severely oversold - strong buying opportunity")
        elif rsi < 30:
            analysis.append(f"💡 RSI at {rsi:.1f} suggests oversold conditions - potential buying opportunity")
        elif 40 <= rsi <= 60:
            analysis.append(f"⚖️ RSI at {rsi:.1f} indicates balanced momentum")
        else:
            bias = "bullish" if rsi > 50 else "bearish"
            analysis.append(f"📊 RSI at {rsi:.1f} shows {bias} bias")

        if latest["Close"] > sma_20 > sma_50:
            analysis.append("📈 Strong bullish alignment - price above both 20 and 50-day MAs")
        elif latest["Close"] < sma_20 < sma_50:
            analysis.append("📉 Bearish trend confirmed - price below key moving averages")
        elif latest["Close"] > sma_20 and sma_20 < sma_50:
            analysis.append("🔄 Mixed signals - short-term bullish but longer-term bearish")
        else:
            analysis.append("➡️ Consolidation phase - awaiting directional breakout")

        if not pd.isna(bb_upper) and latest["Close"] > bb_upper:
            analysis.append("📊 Price trading above upper Bollinger Band - potential overbought")
        elif not pd.isna(bb_lower) and latest["Close"] < bb_lower:
            analysis.append("📊 Price near lower Bollinger Band - potential oversold bounce")

        if macd > macd_signal and macd > 0:
            analysis.append("⚡ MACD shows strong bullish momentum")
        elif macd < macd_signal and macd < 0:
            analysis.append("⚡ MACD indicates bearish momentum")
        elif macd > macd_signal:
            analysis.append("⚡ MACD bullish crossover - momentum improving")
        else:
            analysis.append("⚡ MACD bearish crossover - momentum weakening")

        if volume_ratio > 2:
            analysis.append("🔥 Exceptional volume surge confirms strong conviction")
        elif volume_ratio > 1.5:
            analysis.append("📊 High volume validates price movement")
        elif volume_ratio < 0.5:
            analysis.append("📊 Below-average volume suggests weak conviction")
        else:
            analysis.append("📊 Normal volume levels")

        market_cap = info.get("marketCap", 0) if info else 0
        if market_cap:
            if market_cap > 200e9:
                analysis.append("🏢 Large-cap stability with lower volatility expected")
            elif market_cap > 10e9:
                analysis.append("🏢 Mid-cap stock with balanced growth-stability profile")
            else:
                analysis.append("🏢 Small-cap stock with higher growth potential and volatility")

        return analysis


def create_advanced_chart(data, symbol):
    """Create advanced candlestick chart with technical indicators."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{symbol} Price Action & Moving Averages',
            'Volume', 'MACD', 'RSI & Stochastic',
        ),
        row_heights=[0.5, 0.15, 0.2, 0.15],
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
        ),
        row=1, col=1,
    )

    colors = ['#ff9500', '#007aff', '#5856d6']
    mas = [('SMA_20', 'SMA 20'), ('SMA_50', 'SMA 50'), ('SMA_200', 'SMA 200')]

    for i, (ma_col, ma_name) in enumerate(mas):
        if ma_col in data.columns and not data[ma_col].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data[ma_col],
                    line=dict(color=colors[i], width=1.5), name=ma_name,
                ),
                row=1, col=1,
            )

    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['BB_upper'],
                line=dict(color='rgba(128,128,128,0.5)', width=1),
                name='BB Upper', showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['BB_lower'],
                line=dict(color='rgba(128,128,128,0.5)', width=1),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                name='BB Lower', showlegend=False,
            ),
            row=1, col=1,
        )

    volume_colors = [
        '#00ff88' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ff4444'
        for i in range(len(data))
    ]
    fig.add_trace(
        go.Bar(
            x=data.index, y=data['Volume'],
            marker_color=volume_colors, name='Volume', opacity=0.7,
        ),
        row=2, col=1,
    )

    if 'Volume_SMA' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Volume_SMA'],
                line=dict(color='white', width=1), name='Vol SMA',
            ),
            row=2, col=1,
        )

    if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['MACD'],
                line=dict(color='#007aff', width=2), name='MACD',
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['MACD_signal'],
                line=dict(color='#ff9500', width=2), name='Signal',
            ),
            row=3, col=1,
        )
        histogram_colors = [
            '#00ff88' if val >= 0 else '#ff4444'
            for val in data['MACD_histogram']
        ]
        fig.add_trace(
            go.Bar(
                x=data.index, y=data['MACD_histogram'],
                marker_color=histogram_colors, name='Histogram', opacity=0.6,
            ),
            row=3, col=1,
        )

    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['RSI'],
                line=dict(color='#af52de', width=2), name='RSI',
            ),
            row=4, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=4, col=1)

    if 'Stoch_K' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Stoch_K'],
                line=dict(color='#ffcc00', width=1.5), name='Stoch %K',
            ),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Stoch_D'],
                line=dict(color='#ff6600', width=1.5), name='Stoch %D',
            ),
            row=4, col=1,
        )

    fig.update_layout(
        title=f'{symbol} - Complete Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template='plotly_dark',
        font=dict(size=10),
    )

    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    return fig


def create_backtest_chart(data, model_info, analyzer):
    """Out-of-sample direction backtest, strictly aligned via the pipeline."""
    x_test = model_info['x_test']
    y_test = model_info['y_test']
    predicted = pd.Series(
        analyzer.model.predict(analyzer.scaler.transform(x_test)),
        index=x_test.index,
    )
    direction = np.where(
        predicted.values > x_test['Close_lag_1'].values, 1.0, 0.0
    )
    signals = pd.Series(direction, index=x_test.index)
    bt = pipeline.backtest_returns(data, signals)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=(1 + bt['Forward_Return']).cumprod(),
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#007aff', width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=(1 + bt['Strategy_Return']).cumprod(),
            mode='lines',
            name='Model Direction',
            line=dict(color='#00ff88', width=2),
        )
    )
    fig.update_layout(
        title='Out-of-sample Backtest (entry-date aligned, no look-ahead)',
        xaxis_title='Entry Date',
        yaxis_title='Equity (x)',
        template='plotly_dark',
        height=400,
    )
    return fig


def create_performance_metrics(data, symbol):
    """Create performance metrics visualization."""
    local = data.copy()
    local['Daily_Returns'] = local['Close'].pct_change().dropna()
    local['Cumulative_Returns'] = (1 + local['Daily_Returns']).cumprod() - 1

    daily = local['Daily_Returns'].dropna()
    total_return = local['Cumulative_Returns'].iloc[-1] * 100
    volatility = daily.std() * np.sqrt(252) * 100
    sharpe_ratio = (daily.mean() * 252) / (daily.std() * np.sqrt(252))

    max_drawdown = ((local['Close'] / local['Close'].expanding().max()) - 1).min() * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{total_return:.1f}%")
    with col2:
        st.metric("Volatility (Ann.)", f"{volatility:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=local.index,
            y=local['Cumulative_Returns'] * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00ff88', width=2),
        )
    )

    fig.update_layout(
        title=f'{symbol} Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


PERIOD_PRESETS = {
    '1mo': 30,
    '3mo': 91,
    '6mo': 182,
    '1y': 365,
    '2y': 730,
    '5y': 1825,
}

ADJUST_LABELS = {
    'qfq': '前复权 (Forward adjusted)',
    'hfq': '后复权 (Back adjusted)',
    'none': '不复权 (Raw)',
}


def main():
    st.title("🚀 Professional AI Stock Market Dashboard")
    st.markdown("*Advanced technical analysis with machine learning predictions*")

    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.markdown("---")

    popular_stocks = {
        'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Google': 'GOOGL',
        'Amazon': 'AMZN', 'Tesla': 'TSLA', 'NVIDIA': 'NVDA',
        'Meta': 'META', 'Netflix': 'NFLX', 'AMD': 'AMD', 'Intel': 'INTC',
    }

    stock_choice = st.sidebar.selectbox(
        "🏢 Select Stock:",
        options=list(popular_stocks.keys()) + ['Custom'],
        index=0,
    )

    if stock_choice == 'Custom':
        symbol = st.sidebar.text_input(
            "Enter Stock Symbol:", value="AAPL", max_chars=10
        ).upper()
    else:
        symbol = popular_stocks[stock_choice]

    preset = st.sidebar.selectbox(
        "📅 Analysis Period:",
        options=list(PERIOD_PRESETS.keys()),
        index=3,
    )

    today = datetime.today().date()
    default_start = today - timedelta(days=PERIOD_PRESETS[preset])
    date_cols = st.sidebar.columns(2)
    with date_cols[0]:
        start_date = st.date_input("Start", value=default_start)
    with date_cols[1]:
        end_date = st.date_input("End", value=today)

    adjust = st.sidebar.selectbox(
        "🔧 Adjustment (复权):",
        options=list(ADJUST_LABELS.keys()),
        format_func=lambda key: ADJUST_LABELS[key],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Analysis Options")
    show_prediction = st.sidebar.checkbox("🔮 ML Price Prediction", value=True)
    show_technical = st.sidebar.checkbox("📈 Technical Charts", value=True)
    show_performance = st.sidebar.checkbox("📊 Performance Metrics", value=True)
    show_analysis = st.sidebar.checkbox("🧠 AI Market Analysis", value=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        # Force a live re-fetch but keep the LRU snapshot so three failed
        # attempts can still degrade to the last snapshot with a banner.
        st.session_state["force_refresh"] = True
        st.rerun()

    analyzer = StockAnalyzer()

    start_iso = start_date.isoformat()
    # yfinance's end date is exclusive; fetch one extra day while keeping the
    # cache key on the user-requested window so ranges never collide.
    end_fetch = (
        datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
    ).date().isoformat()

    with st.spinner(f"📡 Fetching live data for {symbol}..."):
        try:
            snapshot = analyzer.fetch_stock_data(
                symbol, start_iso, end_fetch, adjust=adjust,
                force_refresh=st.session_state.pop("force_refresh", False),
            )
        except Exception as exc:
            st.error(
                f"❌ Could not fetch data for {symbol} after "
                f"{pipeline.FETCH_ATTEMPTS} attempts: {exc}"
            )
            st.info("💡 Try popular symbols like AAPL, MSFT, GOOGL, TSLA, etc.")
            return

    data = snapshot.data
    info = snapshot.info or {}

    if data is None or data.empty:
        st.error(f"❌ Could not fetch data for {symbol}. Verify the symbol and try again.")
        return

    adjust_text = ADJUST_LABELS.get(snapshot.adjust, snapshot.adjust)
    if snapshot.stale:
        st.warning(
            f"⚠️ 数据新鲜度提示：实时抓取连续 {pipeline.FETCH_ATTEMPTS} 次超时/失败，"
            f"当前显示最近一次缓存快照（{snapshot.ticker} · {snapshot.start} ~ "
            f"{snapshot.end} · {adjust_text}，快照时间 {snapshot.fetched_at_display()}）。"
            f"原因：{snapshot.error or '未知网络错误'}。请稍后点击 “Refresh Data” 重试。"
        )
    else:
        st.caption(
            f"✅ 实时数据 · 快照时间 {snapshot.fetched_at_display()} · {adjust_text}"
        )

    with st.spinner("⚙️ Calculating technical indicators..."):
        data = analyzer.calculate_technical_indicators(data)

    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)

    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = latest_price - prev_price
    price_change_pct = price_change / prev_price * 100

    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"${latest_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)",
        )

    with col2:
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        if avg_volume and avg_volume > 0:
            volume_change = (volume - avg_volume) / avg_volume * 100
            delta_text = f"{volume_change:+.1f}% vs 20d avg"
        else:
            delta_text = "N/A"
        st.metric(label="📊 Volume", value=f"{volume:,.0f}", delta=delta_text)

    with col3:
        if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric(label="⚡ RSI (14)", value=f"{rsi:.1f}", delta=rsi_status)
        else:
            st.metric(label="⚡ RSI (14)", value="N/A")

    with col4:
        if 'SMA_20' in data.columns and not pd.isna(data['SMA_20'].iloc[-1]):
            sma_20 = data['SMA_20'].iloc[-1]
            sma_distance = (latest_price - sma_20) / sma_20 * 100
            st.metric(
                label="📈 vs SMA 20",
                value=f"{sma_distance:+.1f}%",
                delta="Above" if sma_distance > 0 else "Below",
            )
        else:
            st.metric(label="📈 vs SMA 20", value="N/A")

    with col5:
        market_cap = info.get('marketCap', 0)
        if market_cap:
            if market_cap > 1e12:
                cap_display = f"${market_cap/1e12:.2f}T"
            elif market_cap > 1e9:
                cap_display = f"${market_cap/1e9:.1f}B"
            else:
                cap_display = f"${market_cap/1e6:.0f}M"
            st.metric(label="🏢 Market Cap", value=cap_display)
        else:
            st.metric(label="🏢 Market Cap", value="N/A")

    st.markdown("---")

    if show_technical:
        st.subheader("📈 Advanced Technical Analysis")
        with st.spinner("Creating advanced charts..."):
            chart = create_advanced_chart(data, symbol)
            st.plotly_chart(chart, use_container_width=True)

    if show_performance:
        st.subheader("📊 Performance Analysis")
        create_performance_metrics(data, symbol)

    model_info = None
    if show_prediction:
        st.subheader("🔮 Machine Learning Price Prediction")

        col1, col2 = st.columns([1, 1])

        with col1:
            with st.spinner("🤖 Training AI prediction model..."):
                model_info = analyzer.train_prediction_model(data)

            if model_info:
                prediction = analyzer.predict_next_price(model_info)
                current_price = data['Close'].iloc[-1]
                predicted_change = (prediction - current_price) / current_price * 100

                st.success("✅ Model trained successfully!")

                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric(
                        label="🎯 Next Day Prediction",
                        value=f"${prediction:.2f}",
                        delta=f"{predicted_change:+.2f}%",
                    )
                with pred_col2:
                    confidence = model_info['test_score']
                    confidence_level = (
                        "High" if confidence > 0.8
                        else "Medium" if confidence > 0.6
                        else "Low"
                    )
                    st.metric(
                        label="🎲 Model Confidence",
                        value=f"{confidence:.1%}",
                        delta=confidence_level,
                    )

                st.info(
                    f"📈 **Training Accuracy:** {model_info['train_score']:.1%} | "
                    f"**Test Accuracy:** {model_info['test_score']:.1%}"
                )
            else:
                st.warning("⚠️ Insufficient data for reliable ML prediction. Need more historical data.")

        with col2:
            if model_info:
                importance_df = pd.DataFrame(
                    list(model_info['feature_importance'].items()),
                    columns=['Feature', 'Importance'],
                ).sort_values('Importance', ascending=False).head(10)

                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="🔍 Top 10 Most Important Features",
                    template='plotly_dark',
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)

        if model_info:
            with st.spinner("📉 Running aligned out-of-sample backtest..."):
                st.plotly_chart(
                    create_backtest_chart(data, model_info, analyzer),
                    use_container_width=True,
                )

    if show_analysis:
        st.subheader("🧠 AI-Powered Market Analysis")

        with st.spinner("🤖 Generating intelligent market insights..."):
            analysis = analyzer.generate_market_analysis(data, info, symbol)

        for i, insight in enumerate(analysis):
            if i == 0:
                if "🚀" in insight or "🟢" in insight:
                    st.success(insight)
                elif "🔴" in insight or "🔻" in insight:
                    st.error(insight)
                else:
                    st.warning(insight)
            else:
                st.info(insight)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📋 Company Info", "📊 Raw Data", "🔧 Technical Indicators"]
    )

    with tab1:
        if info:
            col1, col2 = st.columns(2)

            with col1:
                st.write("### 🏢 Company Details")
                company_info = {
                    "Company Name": info.get('longName', 'N/A'),
                    "Sector": info.get('sector', 'N/A'),
                    "Industry": info.get('industry', 'N/A'),
                    "Country": info.get('country', 'N/A'),
                    "Website": info.get('website', 'N/A'),
                    "Employees": (
                        f"{info.get('fullTimeEmployees'):,.0f}"
                        if info.get('fullTimeEmployees') else 'N/A'
                    ),
                }
                for key, value in company_info.items():
                    st.write(f"**{key}:** {value}")

            with col2:
                st.write("### 📈 Financial Metrics")

                def _ratio(name):
                    value = info.get(name)
                    return f"{value:.2f}" if value else 'N/A'

                financial_info = {
                    "P/E Ratio": _ratio('trailingPE'),
                    "Forward P/E": _ratio('forwardPE'),
                    "PEG Ratio": _ratio('pegRatio'),
                    "Price to Book": _ratio('priceToBook'),
                    "Dividend Yield": (
                        f"{info.get('dividendYield', 0)*100:.2f}%"
                        if info.get('dividendYield') else 'N/A'
                    ),
                    "Beta": _ratio('beta'),
                    "52W High": (
                        f"${info['fiftyTwoWeekHigh']:.2f}"
                        if info.get('fiftyTwoWeekHigh') else 'N/A'
                    ),
                    "52W Low": (
                        f"${info['fiftyTwoWeekLow']:.2f}"
                        if info.get('fiftyTwoWeekLow') else 'N/A'
                    ),
                }
                for key, value in financial_info.items():
                    st.write(f"**{key}:** {value}")
        else:
            st.warning("Company information not available")

    with tab2:
        st.write("### 📊 Recent Price Data")
        display_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20)
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        st.dataframe(display_data, use_container_width=True)

        csv = display_data.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'{symbol}_stock_data.csv',
            mime='text/csv',
        )

    with tab3:
        st.write("### 🔧 Technical Indicators (Last 10 Days)")

        tech_columns = [
            'Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'MACD_signal', 'BB_upper', 'BB_lower', 'ATR',
        ]
        available_columns = [col for col in tech_columns if col in data.columns]

        if available_columns:
            tech_data = data[available_columns].tail(10)
            tech_data.index = tech_data.index.strftime('%Y-%m-%d')
            st.dataframe(tech_data.round(3), use_container_width=True)
        else:
            st.warning("Technical indicators not available")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 <strong>AI Stock Dashboard</strong> - Professional technical analysis with machine learning</p>
            <p><em>⚠️ This is for educational purposes only. Not financial advice.</em></p>
            <p>📊 Powered by <a href='https://plotly.com' target='_blank'>Plotly</a> and <a href='https://streamlit.io' target='_blank'>Streamlit</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
