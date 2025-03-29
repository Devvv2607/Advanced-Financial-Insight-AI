import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Advanced Financial Insight AI", layout="wide")

# Title and description
st.title("Advanced Financial Insight AI")
st.markdown("Analyze stocks with AI-powered insights and predictions")

# Sidebar for inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
period = st.sidebar.selectbox("Select Time Period", period_options, index=3)

# Function to fetch stock data with proper caching
@st.cache_data(ttl=3600)  # Using modern cache_data with TTL of 1 hour
def get_stock_data(ticker, period):
    """Fetch stock data using yfinance with proper serialization"""
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get historical data (DataFrame is serializable)
        data = stock.history(period=period)
        
        # Extract only necessary information into serializable dictionaries
        info_dict = {}
        for key in ['shortName', 'sector', 'industry', 'marketCap', 'trailingPE', 
                   'forwardPE', 'dividendYield', 'beta', '52WeekChange']:
            if key in stock.info:
                info_dict[key] = stock.info[key]
        
        return data, stock, info_dict
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame(), None, {}

# Main function
def main():
    # Load data
    with st.spinner("Fetching stock data..."):
        data, stock, info = get_stock_data(ticker, period)
    
    if data.empty:
        st.error("No data available. Please check the ticker symbol and try again.")
        return
    
    # Display company info and current metrics
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Company Overview")
        if 'shortName' in info:
            st.write(f"**Name:** {info.get('shortName', 'N/A')}")
        if 'sector' in info:
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        if 'industry' in info:
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        if 'marketCap' in info:
            market_cap = info.get('marketCap', 0)
            st.write(f"**Market Cap:** ${market_cap:,.0f}")
        if 'beta' in info:
            st.write(f"**Beta:** {info.get('beta', 'N/A'):.2f}")
        if 'trailingPE' in info:
            st.write(f"**Trailing P/E:** {info.get('trailingPE', 'N/A'):.2f}")
        if 'forwardPE' in info:
            st.write(f"**Forward P/E:** {info.get('forwardPE', 'N/A'):.2f}")
        if 'dividendYield' in info:
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield:
                st.write(f"**Dividend Yield:** {dividend_yield:.2%}")
    
    with col2:
        st.subheader("Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        ))
        fig.update_layout(
            title=f"{ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=400,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Technical Analysis
    st.subheader("Technical Analysis")
    
    # Calculate moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # RSI calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Display technical indicators
    tab1, tab2, tab3 = st.tabs(["Moving Averages", "RSI", "MACD"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], mode='lines', name='MA200'))
        fig.update_layout(title="Moving Averages", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
        fig.add_shape(type="line", x0=data.index[0], y0=30, x1=data.index[-1], y1=30,
                    line=dict(color="green", width=2, dash="dash"))
        fig.add_shape(type="line", x0=data.index[0], y0=70, x1=data.index[-1], y1=70,
                    line=dict(color="red", width=2, dash="dash"))
        fig.update_layout(title="Relative Strength Index", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], mode='lines', name='Signal'))
        fig.update_layout(title="MACD", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # LSTM Prediction
    st.subheader("AI Price Prediction (LSTM Model)")
    
    # Prepare data for prediction
    prediction_days = 60  # Number of days to look back
    
    # Get the latest data
    df = data[['Close']].copy()
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Create training dataset
    x_train = []
    y_train = []
    
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])
    
    # Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Check if we have enough data
    if len(x_train) > 0:
        with st.spinner("Training AI model (this may take a moment)..."):
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dense(units=25))
            model.add(Dense(units=1))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train the model (with a small number of epochs for demo)
            model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)
            
            # Predict future prices
            future_days = 30
            
            # Create test dataset
            test_data = scaled_data[-prediction_days:]
            x_test = []
            x_test.append(test_data)
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Initialize predictions array
            predictions = []
            
            # Predict next 'future_days' days
            current_batch = x_test[0]
            for _ in range(future_days):
                current_pred = model.predict(np.array([current_batch]), verbose=0)[0]
                predictions.append(current_pred[0])
                # Update batch for next prediction
                current_batch = np.append(current_batch[1:], current_pred)
                current_batch = current_batch.reshape(prediction_days, 1)
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            # Create future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)
            
            # Create and display prediction chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[-100:], y=data['Close'][-100:], mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Prediction', line=dict(dash='dash')))
            fig.update_layout(
                title=f"{ticker} Price Prediction (Next {future_days} Days)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction metrics
            current_price = data['Close'].iloc[-1]
            last_prediction = predictions[-1][0]
            percent_change = ((last_prediction - current_price) / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Predicted Price (30 days)", f"${last_prediction:.2f}")
            col3.metric("Predicted Change", f"{percent_change:.2f}%", f"{percent_change:.2f}%")
            
            st.caption("⚠️ Disclaimer: This prediction is for demonstration purposes only. Financial markets are complex and predictions may not be accurate.")
    else:
        st.warning("Not enough data for prediction. Please select a longer time period.")

if __name__ == "__main__":
    main()
