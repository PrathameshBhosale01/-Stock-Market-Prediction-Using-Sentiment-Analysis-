import pandas as pd
import numpy as np
import datetime
import streamlit as st
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try: nltk.data.find('vader_lexicon')
except LookupError: nltk.download('vader_lexicon')

st.set_page_config(page_title="Stock Decision Assistant", layout="wide")

stock_mapping = {"Reliance Industries": "RELIANCE", "Tata Consultancy Services": "TCS", "Infosys": "INFY", "HDFC Bank": "HDFCBANK", 
    "ICICI Bank": "ICICIBANK", "State Bank of India": "SBIN", "Bajaj Finance": "BAJFINANCE", "Axis Bank": "AXISBANK", 
    "Hindustan Unilever": "HINDUNILVR", "Tata Motors": "TATAMOTORS", "Wipro": "WIPRO", "HCL Technologies": "HCLTECH", 
    "Tech Mahindra": "TECHM", "Sun Pharma": "SUNPHARMA", "Maruti Suzuki": "MARUTI", "Nestle India": "NESTLEIND", "Mahindra & Mahindra": "M&M"}

@st.cache_data(ttl=300)
def fetch_stock_data(symbol):
    try:
        stock = yf.download(symbol + ".NS", start="2015-01-01", end=datetime.datetime.today().strftime('%Y-%m-%d'), interval="1d")
        if stock.empty: return None
        return stock.reset_index()
    except Exception as e: st.error(f"Error fetching data: {e}"); return None

@st.cache_data(ttl=60)
def fetch_realtime_price(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        todays_data = ticker.history(period='1d')
        if todays_data.empty: return None
        return {'last_price': float(todays_data['Close'].iloc[-1]), 'open_price': float(todays_data['Open'].iloc[-1]), 
                'high_price': float(todays_data['High'].iloc[-1]), 'low_price': float(todays_data['Low'].iloc[-1]), 
                'volume': int(todays_data['Volume'].iloc[-1]), 'timestamp': datetime.datetime.now().strftime('%H:%M:%S')}
    except Exception as e: st.error(f"Error fetching real-time data: {e}"); return None

@st.cache_data
def simple_predict(df):
    try:
        # Create a copy to avoid modifying the original dataframe
        df_pred = df.copy()
        
        # Create date feature
        df_pred['Date'] = pd.to_datetime(df_pred['Date'])
        df_pred['Date_ordinal'] = df_pred['Date'].map(datetime.datetime.toordinal)
        
        # Create simple price features
        df_pred['SMA_20'] = df_pred['Close'].rolling(20).mean()
        df_pred['SMA_50'] = df_pred['Close'].rolling(50).mean()
        
        # Drop NaNs
        df_pred = df_pred.dropna()
        
        # Create feature matrix - keeping it simple to avoid numerical issues
        X = df_pred[['Date_ordinal']].values
        y = df_pred['Close'].values
        
        # Scale features to avoid numerical issues
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model on last 90 days
        model = Ridge(alpha=1.0)
        model.fit(X_scaled[-90:], y[-90:])
        
        # Predict future
        today_ord = datetime.datetime.today().toordinal()
        future_days = [0, 1, 7, 30, 90, 180, 365]
        
        future_X = np.array([[today_ord + days] for days in future_days])
        future_X_scaled = scaler.transform(future_X)
        
        predictions = model.predict(future_X_scaled)
        
        # Ensure predictions are valid numbers
        return [float(max(0, p)) for p in predictions]  # Prevent negative prices
    except Exception as e: st.error(f"Prediction error: {e}"); return [0] * 7

@st.cache_data(ttl=3600)
def fetch_news_headlines(company_name):
    try:
        api_key = '965a5187f0e74bd5a8c1adb65d5589f2'
        url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&apiKey={api_key}"
        resp = requests.get(url)
        headlines = []
        if resp.status_code == 200:
            articles = resp.json().get('articles', [])
            headlines = [{'title': a['title'], 'url': a['url'], 'date': a['publishedAt'][:10]} for a in articles[:8]]
        return headlines
    except Exception as e: st.error(f"Error fetching news: {e}"); return []

@st.cache_data
def analyze_sentiment(headlines):
    try:
        if not headlines: return 0
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(h['title'])['compound'] for h in headlines]
        return np.mean(scores) if scores else 0
    except Exception as e: st.error(f"Sentiment analysis error: {e}"); return 0

def calculate_technical_indicators(df):
    if df is None or df.empty: return None
    df_tech = df.copy()
    
    # Moving Averages
    df_tech['MA20'] = df_tech['Close'].rolling(window=20).mean()
    df_tech['MA50'] = df_tech['Close'].rolling(window=50).mean()
    df_tech['MA200'] = df_tech['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df_tech['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Handle division by zero
    rs = gain / loss.replace(0, np.nan).fillna(loss.mean())
    df_tech['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df_tech['EMA12'] = df_tech['Close'].ewm(span=12, adjust=False).mean()
    df_tech['EMA26'] = df_tech['Close'].ewm(span=26, adjust=False).mean()
    df_tech['MACD'] = df_tech['EMA12'] - df_tech['EMA26']
    df_tech['Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
    df_tech['MACD_Hist'] = df_tech['MACD'] - df_tech['Signal']
    
    # Bollinger Bands
    df_tech['MA20_STD'] = df_tech['Close'].rolling(window=20).std()
    df_tech['Upper_Band'] = df_tech['MA20'] + (df_tech['MA20_STD'] * 2)
    df_tech['Lower_Band'] = df_tech['MA20'] - (df_tech['MA20_STD'] * 2)
    
    # Daily Return
    df_tech['Daily_Return'] = df_tech['Close'].pct_change() * 100
    
    return df_tech

def make_investment_decision(df_tech, sentiment, predictions):
    # Initialize scores
    technical_score = 0
    sentiment_score = 0
    prediction_score = 0
    
    # Make sure df_tech has data and handle empty dataframes
    if df_tech is None or df_tech.empty:
        return "HOLD", 0, "Technical: 0, Sentiment: 0.0, Prediction: 0"
    
    # Get last row with indicators (that isn't NaN) - using iloc to avoid Series comparison issues
    df_clean = df_tech.dropna()
    if df_clean.empty:
        return "HOLD", 0, "Technical: 0, Sentiment: 0.0, Prediction: 0"
    
    last_data = df_clean.iloc[-1]
    
    # 1. Technical Analysis Signals - Use scalar comparisons instead of Series comparisons
    try:
        close_val = float(last_data['Close'])
        ma50_val = float(last_data['MA50'])
        ma200_val = float(last_data['MA200'])
        
        if close_val > ma50_val: technical_score += 1
        else: technical_score -= 1
        
        if ma50_val > ma200_val: technical_score += 1
        else: technical_score -= 1
        
        # RSI signals (oversold/overbought)
        rsi_val = float(last_data['RSI'])
        if rsi_val < 30: technical_score += 2  # Oversold - strong buy
        elif rsi_val > 70: technical_score -= 2  # Overbought - strong sell
        
        # MACD signals
        macd_val = float(last_data['MACD'])
        signal_val = float(last_data['Signal'])
        if macd_val > signal_val: technical_score += 1
        else: technical_score -= 1
        
        # Bollinger Bands
        lower_band_val = float(last_data['Lower_Band'])
        upper_band_val = float(last_data['Upper_Band'])
        if close_val < lower_band_val: technical_score += 1  # Below lower band - potential buy
        elif close_val > upper_band_val: technical_score -= 1  # Above upper band - potential sell
    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Error in technical analysis: {e}")
    
    # 2. Sentiment Analysis
    sentiment_score = sentiment * 5  # Scale from -5 to +5
    
    # 3. Price Prediction
    try:
        # Make sure we're getting a scalar value for current price
        current_price = float(df_tech['Close'].iloc[-1])
        pred_tomorrow, pred_week, pred_month = float(predictions[1]), float(predictions[2]), float(predictions[3])
        
        # Short-term prediction (Tomorrow)
        tomorrow_change = (pred_tomorrow / current_price - 1) * 100
        if tomorrow_change > 2: prediction_score += 2
        elif tomorrow_change > 0.5: prediction_score += 1
        elif tomorrow_change < -2: prediction_score -= 2
        elif tomorrow_change < -0.5: prediction_score -= 1
        
        # Medium-term prediction (Week)
        week_change = (pred_week / current_price - 1) * 100
        if week_change > 5: prediction_score += 2
        elif week_change > 1: prediction_score += 1
        elif week_change < -5: prediction_score -= 2
        elif week_change < -1: prediction_score -= 1
        
        # Long-term prediction (Month)
        month_change = (pred_month / current_price - 1) * 100
        if month_change > 10: prediction_score += 2
        elif month_change > 3: prediction_score += 1
        elif month_change < -10: prediction_score -= 2
        elif month_change < -3: prediction_score -= 1
    except (IndexError, ValueError, TypeError, ZeroDivisionError) as e:
        st.error(f"Error in prediction analysis: {e}")
    
    # Calculate total score with weightings
    total_score = (technical_score * 0.4) + (sentiment_score * 0.2) + (prediction_score * 0.4)
    
    # Make decision
    if total_score >= 3:
        return "STRONG BUY", total_score, f"Technical: {technical_score}, Sentiment: {sentiment_score:.1f}, Prediction: {prediction_score}"
    elif total_score >= 1:
        return "BUY", total_score, f"Technical: {technical_score}, Sentiment: {sentiment_score:.1f}, Prediction: {prediction_score}"
    elif total_score > -1:
        return "HOLD", total_score, f"Technical: {technical_score}, Sentiment: {sentiment_score:.1f}, Prediction: {prediction_score}"
    elif total_score > -3:
        return "SELL", total_score, f"Technical: {technical_score}, Sentiment: {sentiment_score:.1f}, Prediction: {prediction_score}"
    else:
        return "STRONG SELL", total_score, f"Technical: {technical_score}, Sentiment: {sentiment_score:.1f}, Prediction: {prediction_score}"

def plot_stock_price_history(df):
    if df is None or df.empty: return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price", line=dict(color='royalblue', width=2)), secondary_y=False)
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker=dict(color='lightgray')), secondary_y=True)
    fig.update_layout(title_text="Stock Price and Volume History", height=500, hovermode="x unified", 
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.update_yaxes(title_text="Price (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    return fig

def plot_decision_dashboard(df_tech, sentiment, decision, score_details):
    if df_tech is None or df_tech.empty: return None
    
    # FIX: Check for required columns safely and use a column we know exists
    try:
        # Use recent data but don't filter on MA200 which might not exist
        df_recent = df_tech.tail(180).copy()
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"colspan": 2}, None]],
            subplot_titles=("Decision", "Sentiment", "Technical Indicators")
        )
        
        # Decision Gauge - More robust parsing of score details
        try:
            parts = score_details.split(",")
            if len(parts) > 0:
                tech_parts = parts[0].split(":")
                if len(tech_parts) > 1:
                    tech_score_str = tech_parts[1].strip()
                    tech_score = float(tech_score_str)
                    normalized_score = (tech_score + 6) / 12  # Normalize from -6 to 6 into 0 to 1
                else:
                    normalized_score = 0.5
            else:
                normalized_score = 0.5
        except (ValueError, IndexError):
            normalized_score = 0.5  # Default to middle if parsing fails
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=normalized_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"DECISION: {decision}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.2], 'color': "red"},
                        {'range': [0.2, 0.4], 'color': "orange"},
                        {'range': [0.4, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "lightgreen"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': normalized_score
                    }
                }
            ),
            row=1, col=1
        )
        
        # Sentiment Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sentiment Score"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, 0], 'color': "lightcoral"},
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment
                    }
                }
            ),
            row=1, col=2
        )
        
        # Technical chart - Only add MA lines if they exist
        if not df_recent.empty:
            fig.add_trace(
                go.Scatter(x=df_recent['Date'], y=df_recent['Close'], name="Price", line=dict(color='blue')),
                row=2, col=1
            )
            
            # Only add if MA50 exists and has non-null values
            if 'MA50' in df_recent.columns and not df_recent['MA50'].isna().all():
                fig.add_trace(
                    go.Scatter(x=df_recent['Date'], y=df_recent['MA50'], name="MA50", line=dict(color='red')),
                    row=2, col=1
                )
            
            # Only add if MA200 exists and has non-null values
            if 'MA200' in df_recent.columns and not df_recent['MA200'].isna().all():
                fig.add_trace(
                    go.Scatter(x=df_recent['Date'], y=df_recent['MA200'], name="MA200", line=dict(color='green')),
                    row=2, col=1
                )
        
        fig.update_layout(height=700, showlegend=True)
        return fig
    except Exception as e:
        st.error(f"Error creating dashboard: {e}")
        return None

st.title("Stock Market Decision Assistant")
st.markdown("---")

selected_company = st.selectbox("📈 Select Company:", list(stock_mapping.keys()))

with st.spinner('Loading data and generating recommendation...'):
    if selected_company:
        selected_symbol = stock_mapping[selected_company]
        stock_data = fetch_stock_data(selected_symbol)
        realtime_data = fetch_realtime_price(selected_symbol)
    else:
        stock_data, realtime_data = None, None
        st.warning("⚡ Please select a company to analyze.")

if stock_data is not None and not stock_data.empty:
    df_tech = calculate_technical_indicators(stock_data)
    predictions = simple_predict(stock_data)
    headlines = fetch_news_headlines(selected_company)
    sentiment = analyze_sentiment(headlines)
    
    try:
        decision, score, score_details = make_investment_decision(df_tech, sentiment, predictions)
        
        st.header("💰 Investment Decision")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader(f"Recommendation: {decision}")
            st.markdown(f"**Score:** {score:.2f}")
            st.markdown(f"**Score Breakdown:** {score_details}")
            
            try:
                if realtime_data:
                    current_price = float(realtime_data['last_price'])
                    previous_close = float(stock_data['Close'].iloc[-1])
                    price_change = current_price - previous_close
                    price_change_pct = (current_price / previous_close - 1) * 100
                    
                    st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
                    st.info(f"Last updated: {realtime_data['timestamp']}")
                else:
                    current_price = float(stock_data['Close'].iloc[-1])
                    if len(stock_data) > 1:
                        previous_price = float(stock_data['Close'].iloc[-2])
                        price_change = current_price - previous_price
                        price_change_pct = (current_price / previous_price - 1) * 100
                        st.metric("Last Close Price", f"₹{current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
                    else:
                        st.metric("Last Close Price", f"₹{current_price:.2f}", "N/A")
            except (IndexError, KeyError, ValueError, TypeError) as e:
                st.error(f"Error calculating price metrics: {e}")
                st.metric("Price", "N/A", "N/A")
            
            try:
                if len(predictions) >= 7:
                    pred_today, pred_tomorrow, pred_week, pred_month, pred_3month, pred_6month, pred_year = [float(p) for p in predictions]
                    
                    st.subheader("🔮 Price Predictions")
                    
                    # Tomorrow
                    if current_price > 0:
                        pred_tom_change = pred_tomorrow - current_price
                        pred_tom_change_pct = (pred_tomorrow / current_price - 1) * 100
                        st.metric("Tomorrow", f"₹{pred_tomorrow:.2f}", f"{pred_tom_change_pct:.2f}%")
                        
                        # Week
                        pred_week_change_pct = (pred_week / current_price - 1) * 100
                        st.metric("1 Week", f"₹{pred_week:.2f}", f"{pred_week_change_pct:.2f}%")
                        
                        # Month
                        pred_month_change_pct = (pred_month / current_price - 1) * 100
                        st.metric("1 Month", f"₹{pred_month:.2f}", f"{pred_month_change_pct:.2f}%")
                        
                        # Year
                        pred_year_change_pct = (pred_year / current_price - 1) * 100
                        st.metric("1 Year", f"₹{pred_year:.2f}", f"{pred_year_change_pct:.2f}%")
                    else:
                        st.warning("Cannot calculate percentage changes - current price is zero or negative")
                else:
                    st.warning("Insufficient prediction data available")
            except (IndexError, ZeroDivisionError, ValueError, TypeError) as e:
                st.error(f"Error displaying predictions: {e}")
        
        with col2:
            decision_dashboard = plot_decision_dashboard(df_tech, sentiment, decision, score_details)
            if decision_dashboard: st.plotly_chart(decision_dashboard, use_container_width=True)
        
        st.subheader("📈 Price History")
        price_chart = plot_stock_price_history(stock_data)
        if price_chart: st.plotly_chart(price_chart, use_container_width=True)
        
        st.subheader("📰 Recent News Headlines")
        
        if headlines:
            for i, news in enumerate(headlines[:5]):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{news['title']}**")
                        st.markdown(f"[Read more]({news['url']})")
                    with col2:
                        sia = SentimentIntensityAnalyzer()
                        headline_score = sia.polarity_scores(news['title'])['compound']
                        
                        if headline_score >= 0.3: st.markdown("**Sentiment:** 🟢 Positive")
                        elif headline_score <= -0.3: st.markdown("**Sentiment:** 🔴 Negative") 
                        else: st.markdown("**Sentiment:** 🟡 Neutral")
                        
                        st.markdown(f"**Date:** {news['date']}")
                    
                    if i < len(headlines) - 1: st.markdown("---")
        else:
            st.warning("⚡ No recent news found for this company.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning("Please try selecting a different company or refreshing the page.")