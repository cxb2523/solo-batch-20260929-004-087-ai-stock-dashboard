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
warnings.filterwarnings('ignore')

from pipeline import (
    ADJUST_LABELS,
    DEFAULT_FEATURE_COLS,
    DataFetcher,
    align_features_target,
    calculate_technical_indicators as compute_indicators,
    make_label_target,
    prepare_ml_features as build_ml_features,
    run_backtest,
)

# Single fetching layer for the whole app: explicit LRU keyed on
# (ticker, start, end, adjust), with stale-snapshot fallback.
# st.cache_resource keeps one instance across Streamlit reruns (the script
# re-runs top-to-bottom on every interaction), so quotes fetched for one
# user action are reused by the next instead of being re-requested.
@st.cache_resource
def get_data_fetcher():
    return DataFetcher()


data_fetcher = get_data_fetcher()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fetch_stock_data(self, symbol, start, end, adjust="raw"):
        """Fetch through the layered cache (one network call per triple)."""
        return data_fetcher.fetch(symbol, start, end, adjust)

    def calculate_technical_indicators(self, data):
        """Indicator layer: sort by date, NaN until each window is full."""
        return compute_indicators(data)

    def prepare_ml_features(self, data):
        return build_ml_features(data)
    
    def train_prediction_model(self, data):
        """Train on the forward (next-bar) price label.

        Warm-up rows with non-ready features are dropped on the feature
        side; only the final bar (no next price) is dropped on the label
        side. No ffill/bfill, so no information leaks across dates.
        """
        df = self.prepare_ml_features(data)
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
        if len(feature_cols) < 5:
            return None

        target = make_label_target(df['Close'])
        X, y, _tail_dropped = align_features_target(df[feature_cols], target)

        if len(X) < 50:
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        last_row = df[feature_cols].dropna().iloc[-1:]
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
            'last_features': last_row,
            'feature_cols': feature_cols,
            'features_frame': df,
        }
    
    def predict_next_price(self, model_info):
        """Predict next trading day price"""
        if model_info is None:
            return None

        last_features_scaled = self.scaler.transform(model_info['last_features'])
        return float(self.model.predict(last_features_scaled)[0])

    def run_model_backtest(self, data, model_info):
        """Backtest model signals with the same forward shift as the label.

        Predicted next-bar price above today's close -> long (1),
        otherwise flat (0). run_backtest dropna's strictly, so the tail
        bar with no forward return is excluded.
        """
        if model_info is None:
            return None
        df = model_info['features_frame']
        feature_cols = model_info['feature_cols']

        target = make_label_target(df['Close'])
        X, _y, _tail = align_features_target(df[feature_cols], target)
        if X.empty:
            return None

        preds = pd.Series(
            self.model.predict(self.scaler.transform(X)), index=X.index
        )
        current = df['Close'].reindex(X.index)
        position = pd.Series(np.where(preds > current, 1.0, 0.0), index=X.index)
        position = position.reindex(df.index)
        return run_backtest(df['Close'], position)
    
    def generate_market_analysis(self, data, info, symbol):
        """Generate AI-powered market analysis"""
        valid = data.dropna(subset=['Close'])
        latest = valid.iloc[-1]
        prev = valid.iloc[-2]
        info = info or {}
        
        # Price movement
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        
        # Technical analysis (NaN-safe: fall back to neutral/price values)
        rsi = latest['RSI'] if 'RSI' in valid.columns and pd.notna(latest.get('RSI')) else 50
        sma_20 = latest['SMA_20'] if 'SMA_20' in valid.columns and pd.notna(latest.get('SMA_20')) else latest['Close']
        sma_50 = latest['SMA_50'] if 'SMA_50' in valid.columns and pd.notna(latest.get('SMA_50')) else latest['Close']
        bb_upper = latest['BB_upper'] if 'BB_upper' in valid.columns and pd.notna(latest.get('BB_upper')) else latest['Close']
        bb_lower = latest['BB_lower'] if 'BB_lower' in valid.columns and pd.notna(latest.get('BB_lower')) else latest['Close']
        
        # Volume analysis
        avg_volume = data['Volume'].rolling(20, min_periods=20).mean().iloc[-1]
        volume_ratio = latest['Volume'] / avg_volume if pd.notna(avg_volume) and avg_volume > 0 else 1
        
        # MACD analysis
        macd = latest['MACD'] if 'MACD' in valid.columns and pd.notna(latest.get('MACD')) else 0
        macd_signal = latest['MACD_signal'] if 'MACD_signal' in valid.columns and pd.notna(latest.get('MACD_signal')) else 0
        
        # Generate analysis
        analysis = []
        
        # Price trend
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
        
        # RSI analysis
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
            analysis.append(f"📊 RSI at {rsi:.1f} shows {('bullish' if rsi > 50 else 'bearish')} bias")
        
        # Moving average analysis
        if latest['Close'] > sma_20 > sma_50:
            analysis.append("📈 Strong bullish alignment - price above both 20 and 50-day MAs")
        elif latest['Close'] < sma_20 < sma_50:
            analysis.append("📉 Bearish trend confirmed - price below key moving averages")
        elif latest['Close'] > sma_20 and sma_20 < sma_50:
            analysis.append("🔄 Mixed signals - short-term bullish but longer-term bearish")
        else:
            analysis.append("➡️ Consolidation phase - awaiting directional breakout")
        
        # Bollinger Bands analysis
        if latest['Close'] > bb_upper:
            analysis.append("📊 Price trading above upper Bollinger Band - potential overbought")
        elif latest['Close'] < bb_lower:
            analysis.append("📊 Price near lower Bollinger Band - potential oversold bounce")
        
        # MACD analysis
        if macd > macd_signal and macd > 0:
            analysis.append("⚡ MACD shows strong bullish momentum")
        elif macd < macd_signal and macd < 0:
            analysis.append("⚡ MACD indicates bearish momentum")
        elif macd > macd_signal:
            analysis.append("⚡ MACD bullish crossover - momentum improving")
        else:
            analysis.append("⚡ MACD bearish crossover - momentum weakening")
        
        # Volume analysis
        if volume_ratio > 2:
            analysis.append("🔥 Exceptional volume surge confirms strong conviction")
        elif volume_ratio > 1.5:
            analysis.append("📊 High volume validates price movement")
        elif volume_ratio < 0.5:
            analysis.append("📊 Below-average volume suggests weak conviction")
        else:
            analysis.append("📊 Normal volume levels")
        
        # Market cap context
        market_cap = info.get('marketCap', 0) or 0
        if market_cap:
            if market_cap > 200e9:  # > 200B
                analysis.append("🏢 Large-cap stability with lower volatility expected")
            elif market_cap > 10e9:  # > 10B
                analysis.append("🏢 Mid-cap stock with balanced growth-stability profile")
            else:
                analysis.append("🏢 Small-cap stock with higher growth potential and volatility")
        
        return analysis

def create_advanced_chart(data, symbol):
    """Create advanced candlestick chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Action & Moving Averages', 'Volume', 'MACD', 'RSI & Stochastic'),
        row_heights=[0.5, 0.15, 0.2, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Moving averages
    colors = ['#ff9500', '#007aff', '#5856d6']
    mas = [('SMA_20', 'SMA 20'), ('SMA_50', 'SMA 50'), ('SMA_200', 'SMA 200')]
    
    for i, (ma_col, ma_name) in enumerate(mas):
        if ma_col in data.columns and not data[ma_col].isna().all():
            fig.add_trace(
                go.Scatter(x=data.index, y=data[ma_col], 
                          line=dict(color=colors[i], width=1.5), name=ma_name),
                row=1, col=1
            )
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_upper'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Upper',
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_lower'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Lower',
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                      showlegend=False),
            row=1, col=1
        )
    
    # Volume
    volume_colors = ['#00ff88' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ff4444' 
                    for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], 
              marker_color=volume_colors, name='Volume', opacity=0.7),
        row=2, col=1
    )
    
    if 'Volume_SMA' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Volume_SMA'], 
                      line=dict(color='white', width=1), name='Vol SMA'),
            row=2, col=1
        )
    
    # MACD
    if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], 
                      line=dict(color='#007aff', width=2), name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_signal'], 
                      line=dict(color='#ff9500', width=2), name='Signal'),
            row=3, col=1
        )
        
        histogram_colors = ['#00ff88' if val >= 0 else '#ff4444' for val in data['MACD_histogram']]
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_histogram'], 
                  marker_color=histogram_colors, name='Histogram', opacity=0.6),
            row=3, col=1
        )
    
    # RSI and Stochastic
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], 
                      line=dict(color='#af52de', width=2), name='RSI'),
            row=4, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=4, col=1)
    
    if 'Stoch_K' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_K'], 
                      line=dict(color='#ffcc00', width=1.5), name='Stoch %K'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_D'], 
                      line=dict(color='#ff6600', width=1.5), name='Stoch %D'),
            row=4, col=1
        )
    
    fig.update_layout(
        title=f'{symbol} - Complete Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template='plotly_dark',
        font=dict(size=10)
    )
    
    # Remove x-axis labels from all but bottom subplot
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    return fig

def create_performance_metrics(data, symbol):
    """Create performance metrics visualization"""
    # Calculate returns on a sorted copy (never mutate the chart frame)
    df = data.sort_index().copy()
    df['Daily_Returns'] = df['Close'].pct_change()
    df['Cumulative_Returns'] = (1 + df['Daily_Returns']).cumprod() - 1
    
    # Performance metrics
    daily = df['Daily_Returns'].dropna()
    total_return = df['Cumulative_Returns'].iloc[-1] * 100
    volatility = daily.std() * np.sqrt(252) * 100  # Annualized
    sharpe_ratio = (daily.mean() * 252) / (daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    
    max_drawdown = ((df['Close'] / df['Close'].expanding().max()) - 1).min() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_return:.1f}%")
    with col2:
        st.metric("Volatility (Ann.)", f"{volatility:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    # Cumulative returns chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Cumulative_Returns'] * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00ff88', width=2)
        )
    )
    
    fig.update_layout(
        title=f'{symbol} Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_backtest_chart(backtest, symbol):
    """Plot strategy vs buy-and-hold equity from the aligned backtest."""
    if backtest is None or backtest.empty:
        st.info("Not enough aligned bars to run the backtest.")
        return

    total_ret = (backtest['equity'].iloc[-1] - 1) * 100
    bench_ret = (backtest['benchmark'].iloc[-1] - 1) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Strategy Return", f"{total_ret:.2f}%")
    c2.metric("Buy & Hold Return", f"{bench_ret:.2f}%")
    c3.metric("Bars Traded", f"{int((backtest['position'] != 0).sum())} / {len(backtest)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=backtest.index, y=backtest['equity'], mode='lines',
        name='ML Strategy', line=dict(color='#00ff88', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=backtest.index, y=backtest['benchmark'], mode='lines',
        name='Buy & Hold', line=dict(color='#007aff', width=2, dash='dash')
    ))
    fig.update_layout(
        title=f'{symbol} Backtest - Equity Curve (strictly aligned, no look-ahead)',
        xaxis_title='Date', yaxis_title='Growth of $1',
        template='plotly_dark', height=380
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit App
def main():
    st.title("🚀 Professional AI Stock Market Dashboard")
    st.markdown("*Advanced technical analysis with machine learning predictions*")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Stock selection with popular choices
    popular_stocks = {
        'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Google': 'GOOGL', 
        'Amazon': 'AMZN', 'Tesla': 'TSLA', 'NVIDIA': 'NVDA',
        'Meta': 'META', 'Netflix': 'NFLX', 'AMD': 'AMD', 'Intel': 'INTC'
    }
    
    stock_choice = st.sidebar.selectbox(
        "🏢 Select Stock:",
        options=list(popular_stocks.keys()) + ['Custom'],
        index=0
    )
    
    if stock_choice == 'Custom':
        symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL", max_chars=10).upper()
    else:
        symbol = popular_stocks[stock_choice]
    
    # Explicit date range (both ends are part of the cache key)
    st.sidebar.subheader("📅 Date Range")
    range_preset = st.sidebar.selectbox(
        "Quick range:",
        options=['Custom', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=4
    )
    today = datetime.now().date()
    preset_days = {'1mo': 30, '3mo': 91, '6mo': 182, '1y': 365, '2y': 730, '5y': 1825}
    default_start = today - timedelta(days=preset_days.get(range_preset, 365))
    start_date = st.sidebar.date_input("Start date:", value=default_start)
    end_date = st.sidebar.date_input("End date:", value=today)
    if start_date >= end_date:
        st.sidebar.error("Start date must be earlier than end date.")

    # Adjustment mode - the third element of the cache key
    adjust = st.sidebar.selectbox(
        "💱 Adjustment:",
        options=list(ADJUST_LABELS.keys()),
        format_func=lambda k: ADJUST_LABELS[k],
        index=3
    )

    st.sidebar.markdown("---")

    # Analysis options
    st.sidebar.subheader("🔧 Analysis Options")
    show_prediction = st.sidebar.checkbox("🔮 ML Price Prediction", value=True)
    show_technical = st.sidebar.checkbox("📈 Technical Charts", value=True)
    show_performance = st.sidebar.checkbox("📊 Performance Metrics", value=True)
    show_analysis = st.sidebar.checkbox("🧠 AI Market Analysis", value=True)

    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        data_fetcher.clear(symbol)
        st.rerun()

    # Initialize analyzer
    analyzer = StockAnalyzer()

    # Fetch through the layered LRU (key = ticker + start + end + adjust)
    with st.spinner(f"📡 Fetching live data for {symbol}..."):
        fetch_result = analyzer.fetch_stock_data(symbol, start_date, end_date, adjust)

    if not fetch_result.ok:
        st.error(f"❌ Could not fetch data for {symbol} after {fetch_result.attempts} attempts.")
        if fetch_result.error:
            st.caption(f"Last error: {fetch_result.error}")
        st.info("💡 Try popular symbols like AAPL, MSFT, GOOGL, TSLA, etc.")
        return

    raw_data, info = fetch_result.data, fetch_result.info

    # Indicator layer: sort by date, NaN where windows are insufficient
    with st.spinner("⚙️ Calculating technical indicators..."):
        data = analyzer.calculate_technical_indicators(raw_data)

    # Data freshness banner (stale-snapshot degradation shows at the top)
    freshness = fetch_result.freshness_text()
    if freshness:
        if fetch_result.stale:
            st.warning(freshness)
        elif fetch_result.cached:
            st.caption(freshness)
        else:
            st.success(freshness)
    
    # Main dashboard header
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"${latest_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20, min_periods=20).mean().iloc[-1]
        if pd.notna(avg_volume) and avg_volume > 0:
            volume_change = ((volume - avg_volume) / avg_volume) * 100
            volume_delta = f"{volume_change:+.1f}% vs 20d avg"
        else:
            volume_delta = "N/A vs 20d avg"
        st.metric(label="📊 Volume", value=f"{volume:,.0f}", delta=volume_delta)
    
    with col3:
        if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric(
                label="⚡ RSI (14)",
                value=f"{rsi:.1f}",
                delta=rsi_status
            )
        else:
            st.metric(label="⚡ RSI (14)", value="N/A")
    
    with col4:
        if 'SMA_20' in data.columns and not pd.isna(data['SMA_20'].iloc[-1]):
            sma_20 = data['SMA_20'].iloc[-1]
            sma_distance = ((latest_price - sma_20) / sma_20) * 100
            st.metric(
                label="📈 vs SMA 20",
                value=f"{sma_distance:+.1f}%",
                delta="Above" if sma_distance > 0 else "Below"
            )
        else:
            st.metric(label="📈 vs SMA 20", value="N/A")
    
    with col5:
        market_cap = info.get('marketCap', 0) or 0
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
    
    # Advanced Chart
    if show_technical:
        st.subheader("📈 Advanced Technical Analysis")
        with st.spinner("Creating advanced charts..."):
            chart = create_advanced_chart(data, symbol)
            st.plotly_chart(chart, use_container_width=True)
    
    # Performance Metrics
    if show_performance:
        st.subheader("📊 Performance Analysis")
        create_performance_metrics(data, symbol)
    
    # ML Prediction
    if show_prediction:
        st.subheader("🔮 Machine Learning Price Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.spinner("🤖 Training AI prediction model..."):
                model_info = analyzer.train_prediction_model(data)
            
            if model_info:
                prediction = analyzer.predict_next_price(model_info)
                current_price = data['Close'].iloc[-1]
                predicted_change = ((prediction - current_price) / current_price) * 100
                
                st.success("✅ Model trained successfully!")
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric(
                        label="🎯 Next Day Prediction",
                        value=f"${prediction:.2f}",
                        delta=f"{predicted_change:+.2f}%"
                    )
                
                with pred_col2:
                    confidence = model_info['test_score']
                    confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric(
                        label="🎲 Model Confidence",
                        value=f"{confidence:.1%}",
                        delta=confidence_level
                    )
                
                # Model performance
                st.info(f"📈 **Training Accuracy:** {model_info['train_score']:.1%} | **Test Accuracy:** {model_info['test_score']:.1%}")
            else:
                st.warning("⚠️ Insufficient data for reliable ML prediction. Need more historical data.")
        
        with col2:
            if model_info:
                # Feature importance
                importance_df = pd.DataFrame(
                    list(model_info['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="🔍 Top 10 Most Important Features",
                    template='plotly_dark'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)

        # Backtest uses the identical forward shift as the training label;
        # the series is strictly dropna'd (last bar excluded).
        if model_info:
            with st.spinner("📊 Running look-ahead-free backtest..."):
                backtest = analyzer.run_model_backtest(data, model_info)
            create_backtest_chart(backtest, symbol)
    
    # AI Market Analysis
    if show_analysis:
        st.subheader("🧠 AI-Powered Market Analysis")
        
        with st.spinner("🤖 Generating intelligent market insights..."):
            analysis = analyzer.generate_market_analysis(data, info, symbol)
        
        # Display analysis in an attractive format
        for i, insight in enumerate(analysis):
            if i == 0:  # First insight (price movement) gets special treatment
                if "🚀" in insight or "🟢" in insight:
                    st.success(insight)
                elif "🔴" in insight or "🔻" in insight:
                    st.error(insight)
                else:
                    st.warning(insight)
            else:
                st.info(insight)
    
    # Additional Analysis Tabs
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Company Info", "📊 Raw Data", "🔧 Technical Indicators"])
    
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
                    "Employees": f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else 'N/A'
                }
                
                for key, value in company_info.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.write("### 📈 Financial Metrics")
                financial_info = {
                    "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A',
                    "Forward P/E": f"{info.get('forwardPE', 'N/A'):.2f}" if info.get('forwardPE') else 'N/A',
                    "PEG Ratio": f"{info.get('pegRatio', 'N/A'):.2f}" if info.get('pegRatio') else 'N/A',
                    "Price to Book": f"{info.get('priceToBook', 'N/A'):.2f}" if info.get('priceToBook') else 'N/A',
                    "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
                    "Beta": f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else 'N/A',
                    "52W High": f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                    "52W Low": f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A'
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
        
        # Download option
        csv = display_data.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'{symbol}_stock_data.csv',
            mime='text/csv'
        )
    
    with tab3:
        st.write("### 🔧 Technical Indicators (Last 10 Days)")
        
        tech_columns = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR']
        available_columns = [col for col in tech_columns if col in data.columns]
        
        if available_columns:
            tech_data = data[available_columns].tail(10)
            tech_data.index = tech_data.index.strftime('%Y-%m-%d')
            st.dataframe(tech_data.round(3), use_container_width=True)
        else:
            st.warning("Technical indicators not available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 <strong>AI Stock Dashboard</strong> - Professional technical analysis with machine learning</p>
            <p><em>⚠️ This is for educational purposes only. Not financial advice.</em></p>
            <p>Built with ❤️ by <a href='https://erikthiart.com' target='_blank'>Erik Thiart</a></p>
            <p>📊 Powered by <a href='https://plotly.com' target='_blank'>Plotly</a> and <a href='https://streamlit.io' target='_blank'>Streamlit</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
