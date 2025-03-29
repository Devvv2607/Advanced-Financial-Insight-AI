import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Advanced Financial Insight AI",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("📊 Advanced Financial Insight AI")
st.markdown("Analyze stock performance, volatility, and financial metrics with AI-powered insights.")

# Sidebar for user inputs
with st.sidebar:
    st.header("Stock Selection")
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, NVDA):", "AAPL").upper()
    
    st.header("Time Period")
    period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
    period = st.selectbox("Select Time Period:", period_options, index=3)
    
    st.header("Analysis Options")
    show_volatility = st.checkbox("Show Volatility Analysis", value=True)
    show_volume = st.checkbox("Show Volume Analysis", value=True)
    show_financial_metrics = st.checkbox("Show Financial Metrics", value=True)
    show_earnings = st.checkbox("Show Earnings Data", value=True)

# Function to get stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        info = stock.info
        return data, stock, info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None

# Main function to fetch data
data, stock, info = get_stock_data(ticker, period)

if data is not None and not data.empty:
    # Show basic stock info
    col1, col2, col3 = st.columns(3)
    
    try:
        with col1:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}", 
                     f"{((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%")
        with col2:
            st.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}", "")
        with col3:
            st.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}", "")
    except Exception as e:
        st.warning(f"Could not display some metrics: {e}")
    
    # Basic info table
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Company Information")
        info_data = {
            "Name": info.get('longName', ticker),
            "Sector": info.get('sector', 'N/A'),
            "Industry": info.get('industry', 'N/A'),
            "Market Cap": f"${info.get('marketCap', 0)/1000000000:.2f}B" if info.get('marketCap') else 'N/A',
            "P/E Ratio": f"{info.get('trailingPE', 'N/A')}",
            "Dividend Yield": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
        }
        
        st.table(pd.DataFrame(list(info_data.items()), columns=['Metric', 'Value']))
    
    # Stock price chart
    st.subheader("📈 Stock Price History")
    
    # Create Plotly figure for stock price
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='rgba(0, 128, 0, 0.8)', width=2)
    ))
    
    # Add moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='20-Day MA', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50-Day MA', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], mode='lines', name='200-Day MA', line=dict(color='red', width=1)))
    
    fig.update_layout(
        title=f"{info.get('longName', ticker)} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility Analysis
    if show_volatility:
        st.subheader("📊 Stock Price and Volatility")
        
        try:
            # Calculate daily returns
            data['Daily_Return'] = data['Close'].pct_change() * 100
            
            # Calculate volatility (20-day rolling standard deviation of returns)
            data['Volatility'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252) * 100
            
            # Create volatility chart
            vol_fig = go.Figure()
            
            vol_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Volatility'],
                mode='lines',
                name='Volatility (Annualized)',
                line=dict(color='red', width=2)
            ))
            
            vol_fig.update_layout(
                title=f"{ticker} Volatility (20-Day Rolling)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                hovermode="x unified",
                template="plotly_dark"
            )
            
            st.plotly_chart(vol_fig, use_container_width=True)
            
            # Calculate drawdowns
            data['Cumulative_Return'] = (1 + data['Daily_Return'] / 100).cumprod()
            data['Rolling_Max'] = data['Cumulative_Return'].cummax()
            data['Drawdown'] = (data['Cumulative_Return'] / data['Rolling_Max'] - 1) * 100
            
            # Create drawdown chart
            dd_fig = go.Figure()
            
            dd_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color='firebrick', width=2),
                fill='tozeroy',
                fillcolor='rgba(178, 34, 34, 0.3)'
            ))
            
            dd_fig.update_layout(
                title=f"{ticker} Drawdown Analysis",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode="x unified",
                template="plotly_dark",
                yaxis=dict(range=[data['Drawdown'].min() * 1.1, 5])
            )
            
            st.plotly_chart(dd_fig, use_container_width=True)
            
            # Risk metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Annualized Volatility", f"{data['Volatility'].dropna().mean():.2f}%")
            with col2:
                st.metric("Max Drawdown", f"{data['Drawdown'].min():.2f}%")
            with col3:
                sharpe = data['Daily_Return'].mean() / data['Daily_Return'].std() * np.sqrt(252)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            with col4:
                sortino = data['Daily_Return'].mean() / data.loc[data['Daily_Return'] < 0, 'Daily_Return'].std() * np.sqrt(252)
                st.metric("Sortino Ratio", f"{sortino:.2f}")
            
        except Exception as e:
            st.error(f"Error creating volatility chart: {str(e)}")
    
    # Volume Analysis
    if show_volume:
        st.subheader("🔄 Trading Volume Analysis")
        
        try:
            # Volume chart
            vol_chart = go.Figure()
            
            vol_chart.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker=dict(color='rgba(58, 71, 80, 0.8)')
            ))
            
            vol_chart.update_layout(
                title=f"{ticker} Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                hovermode="x unified",
                template="plotly_dark"
            )
            
            # Calculate 20-day average volume
            data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
            
            vol_chart.add_trace(go.Scatter(
                x=data.index,
                y=data['Volume_MA20'],
                mode='lines',
                name='20-Day Average Volume',
                line=dict(color='orange', width=2)
            ))
            
            st.plotly_chart(vol_chart, use_container_width=True)
            
            # Volume metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Daily Volume", f"{data['Volume'].mean():,.0f}")
            with col2:
                st.metric("Recent Volume Trend", 
                        f"{(data['Volume'].iloc[-5:].mean() / data['Volume'].iloc[-20:-5].mean() - 1) * 100:.2f}%",
                        f"{(data['Volume'].iloc[-5:].mean() - data['Volume'].iloc[-20:-5].mean()):,.0f}")
            with col3:
                st.metric("Volume Volatility", f"{data['Volume'].pct_change().std() * 100:.2f}%")
            
        except Exception as e:
            st.error(f"Error creating volume chart: {str(e)}")
    
    # Financial Metrics
    if show_financial_metrics:
        st.subheader("💰 Key Financial Metrics")
        
        try:
            # Get financial data
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cashflow
            
            if not balance_sheet.empty and not income_stmt.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Income Statement Metrics")
                    
                    # Extract and display income statement metrics
                    if "Total Revenue" in income_stmt.index:
                        revenue = income_stmt.loc["Total Revenue"].iloc[0]
                        rev_growth = (income_stmt.loc["Total Revenue"].iloc[0] / income_stmt.loc["Total Revenue"].iloc[1] - 1) * 100 if len(income_stmt.columns) > 1 else 0
                        st.metric("Revenue (TTM)", f"${revenue/1000000000:.2f}B", f"{rev_growth:.2f}% YoY")
                    
                    if "Net Income" in income_stmt.index:
                        net_income = income_stmt.loc["Net Income"].iloc[0]
                        ni_growth = (income_stmt.loc["Net Income"].iloc[0] / income_stmt.loc["Net Income"].iloc[1] - 1) * 100 if len(income_stmt.columns) > 1 else 0
                        st.metric("Net Income (TTM)", f"${net_income/1000000000:.2f}B", f"{ni_growth:.2f}% YoY")
                    
                    if "EBITDA" in income_stmt.index:
                        ebitda = income_stmt.loc["EBITDA"].iloc[0]
                        st.metric("EBITDA (TTM)", f"${ebitda/1000000000:.2f}B")
                    
                    if "Total Revenue" in income_stmt.index and "Net Income" in income_stmt.index:
                        profit_margin = net_income / revenue * 100
                        st.metric("Profit Margin", f"{profit_margin:.2f}%")
                
                with col2:
                    st.subheader("Balance Sheet Metrics")
                    
                    # Extract and display balance sheet metrics
                    if "Total Assets" in balance_sheet.index:
                        assets = balance_sheet.loc["Total Assets"].iloc[0]
                        st.metric("Total Assets", f"${assets/1000000000:.2f}B")
                    
                    if "Total Debt" in balance_sheet.index:
                        debt = balance_sheet.loc["Total Debt"].iloc[0]
                        st.metric("Total Debt", f"${debt/1000000000:.2f}B")
                    
                    if "Total Assets" in balance_sheet.index and "Total Debt" in balance_sheet.index:
                        debt_to_assets = debt / assets * 100
                        st.metric("Debt-to-Assets Ratio", f"{debt_to_assets:.2f}%")
                    
                    if "Total Cash" in balance_sheet.index:
                        cash = balance_sheet.loc["Total Cash"].iloc[0]
                        st.metric("Cash & Equivalents", f"${cash/1000000000:.2f}B")
            else:
                st.warning("No financial data available for this stock.")
            
        except Exception as e:
            st.error(f"Error retrieving financial metrics: {str(e)}")
    
    # Earnings Analysis
    if show_earnings:
        st.subheader("📝 Earnings Analysis")
        
        try:
            # Get earnings data
            earnings = stock.earnings
            
            if not earnings.empty:
                # Display earnings table
                st.subheader("Historical Earnings")
                st.dataframe(earnings)
                
                # Create earnings chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=earnings.index,
                    y=earnings['Revenue'],
                    name='Revenue',
                    marker_color='rgb(55, 83, 109)'
                ))
                
                fig.add_trace(go.Bar(
                    x=earnings.index,
                    y=earnings['Earnings'],
                    name='Earnings',
                    marker_color='rgb(26, 118, 255)'
                ))
                
                fig.update_layout(
                    title=f"{ticker} Historical Earnings",
                    xaxis_title="Year",
                    yaxis_title="Amount (USD)",
                    barmode='group',
                    hovermode="x unified",
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Get earnings calendar
                earnings_calendar = stock.calendar
                
                if earnings_calendar:
                    st.subheader("Upcoming Earnings Date")
                    next_earnings = earnings_calendar.get('Earnings Date', 'No upcoming earnings date found')
                    
                    if isinstance(next_earnings, datetime):
                        days_until = (next_earnings - datetime.now()).days
                        st.info(f"Next Earnings Date: {next_earnings.strftime('%B %d, %Y')} ({days_until} days from now)")
                    else:
                        st.info("No upcoming earnings date found")
            else:
                st.warning("No earnings data found for NVDA")
                
        except Exception as e:
            st.error(f"Error retrieving earnings data: {str(e)}")
    
    # Final section - Summary and AI Analysis
    st.subheader("💡 Market Summary")
    market_summary = f"""
    **{info.get('longName', ticker)} ({ticker})** is currently trading at **${data['Close'].iloc[-1]:.2f}**. 
    
    The stock has moved {(data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100:.2f}% over the selected period.
    
    **Key Observations:**
    - Average daily volume: {data['Volume'].mean():,.0f} shares
    - Current volatility: {data['Volatility'].dropna().iloc[-1]:.2f}%
    - Maximum drawdown: {data['Drawdown'].min():.2f}%
    
    Based on technical indicators, the stock is currently trading 
    {('above' if data['Close'].iloc[-1] > data['MA50'].iloc[-1] else 'below')} its 50-day moving average and
    {('above' if data['Close'].iloc[-1] > data['MA200'].iloc[-1] else 'below')} its 200-day moving average.
    """
    
    st.markdown(market_summary)

else:
    st.error(f"Could not retrieve data for ticker: {ticker}. Please check if the symbol is correct.")

# Footer
st.markdown("---")
st.caption("Disclaimer: This app is for informational purposes only and does not constitute investment advice.")
