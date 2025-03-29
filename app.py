import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from groq import Groq
import os
import traceback
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Streamlit App Configuration
st.set_page_config(
    page_title="Advanced Financial Insight AI",
    page_icon="💹",
    layout="wide"
)

# Robust Groq Client Initialization
def get_groq_client():
    try:
        # Try Streamlit secrets first
        groq_api_key = None
        
        # Check Streamlit secrets
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
        
        # Check environment variables
        if not groq_api_key:
            groq_api_key = os.getenv("GROQ_API_KEY")
        
        # If no key found, prompt user
        if not groq_api_key:
            groq_api_key = st.sidebar.text_input(
                "Enter Groq API Key", 
                type="password", 
                help="Get your API key from https://console.groq.com"
            )
        
        # Final validation
        if not groq_api_key:
            st.sidebar.warning("🚨 Groq API Key is required for AI analysis.")
            return None
        
        return Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Groq Client Error: {e}")
        st.error(traceback.format_exc())
        return None

# Fetch Stock Information
def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        stock_data = {
            "Company Name": info.get('longName', 'N/A'),
            "Current Price": f"${info.get('currentPrice', 'N/A'):.2f}",
            "Market Cap": f"${info.get('marketCap', 'N/A'):,}",
            "PE Ratio": info.get('trailingPE', 'N/A'),
            "Dividend Yield": f"{info.get('dividendYield', 'N/A'):.2%}",
            "52 Week High": f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}",
            "52 Week Low": f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}",
            "Sector": info.get('sector', 'N/A'),
            "Industry": info.get('industry', 'N/A')
        }
        
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock information: {e}")
        return None

# Fetch Historical Stock Data and Calculate Volatility
def get_stock_volatility(symbol, period='1y'):
    try:
        stock_data = yf.download(symbol, period=period)
        
        if stock_data.empty:
            st.error(f"No historical data found for {symbol}")
            return None
        
        stock_data['Daily Returns'] = stock_data['Close'].pct_change()
        stock_data['Rolling Volatility'] = stock_data['Daily Returns'].rolling(window=30).std() * (252 ** 0.5)
        
        return stock_data
    except Exception as e:
        st.error(f"Error fetching historical stock data: {e}")
        return None

# Fetch Historical PE Ratio
def get_historical_pe_ratio(symbol, period='1y'):
    try:
        stock = yf.Ticker(symbol)
        earnings = stock.quarterly_earnings
        
        if earnings is None or earnings.empty:
            st.error(f"No earnings data found for {symbol}")
            return None
        
        pe_ratios = []
        dates = []
        
        for index, row in earnings.iterrows():
            historical_data = yf.download(symbol, start=index-pd.Timedelta(days=7), end=index+pd.Timedelta(days=7))
            
            if not historical_data.empty:
                avg_price = historical_data['Close'].mean()
                
                if row['Earnings'] != 0:
                    pe_ratio = avg_price / row['Earnings']
                    pe_ratios.append(pe_ratio)
                    dates.append(index)
        
        pe_df = pd.DataFrame({
            'Date': dates,
            'PE Ratio': pe_ratios
        }).set_index('Date')
        
        return pe_df
    except Exception as e:
        st.error(f"Error fetching historical PE Ratio: {e}")
        return None

# Create Price and Volatility Chart
def create_price_volatility_chart(stock_data):
    try:
        fig = plt.Figure()
        
        fig.add_trace(
            plt.Scatter(
                x=stock_data.index, 
                y=stock_data['Close'], 
                name='Stock Price', 
                line=dict(color='blue'),
                yaxis='y1'
            )
        )
        
        fig.add_trace(
            plt.Scatter(
                x=stock_data.index, 
                y=stock_data['Rolling Volatility'], 
                name='30-Day Volatility', 
                line=dict(color='red'),
                yaxis='y2'
            )
        )
        
        fig.update_layout(
            title=f'Stock Price and Volatility',
            xaxis_title='Date',
            yaxis_title='Stock Price',
            yaxis2=dict(
                title='Annualized Volatility',
                overlaying='y',
                side='right'
            ),
            height=400,
            legend=dict(x=0, y=1.1, orientation='h')
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating volatility chart: {e}")
        return None

# Create PE Ratio Chart
def create_pe_ratio_chart(pe_data):
    try:
        if pe_data is None or pe_data.empty:
            st.warning("No PE Ratio data available")
            return None

        fig = plt.Figure()
        
        fig.add_trace(
            plt.Scatter(
                x=pe_data.index, 
                y=pe_data['PE Ratio'], 
                name='PE Ratio', 
                line=dict(color='green')
            )
        )
        
        fig.update_layout(
            title='Historical PE Ratio',
            xaxis_title='Date',
            yaxis_title='PE Ratio',
            height=400,
            legend=dict(x=0, y=1.1, orientation='h')
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating PE Ratio chart: {e}")
        return None

# Fetch News
def get_duckduckgo_news(symbol, limit=5):
    try:
        with DDGS() as ddgs:
            news_results = list(ddgs.news(f"{symbol} stock recent news", max_results=limit))
            
            formatted_news = [
                {
                    "title": result.get('title', 'N/A'),
                    "link": result.get('url', ''),
                    "publisher": result.get('source', 'N/A'),
                    "source": "DuckDuckGo"
                } for result in news_results
            ]
            
            return formatted_news
    except Exception as e:
        st.warning(f"DuckDuckGo news search error: {e}")
        return []

# Generate AI Analysis
def generate_ai_analysis(stock_info, news, query_type):
    client = get_groq_client()
    
    # Fallback if no client
    if not client:
        return (
            "⚠️ AI Analysis Unavailable\n\n"
            "To enable AI insights:\n"
            "1. Get a Groq API Key from https://console.groq.com\n"
            "2. Enter the API key in the sidebar input\n\n"
            f"Stock Overview:\n"
            f"Company: {stock_info.get('Company Name', 'N/A')}\n"
            f"Price: {stock_info.get('Current Price', 'N/A')}\n"
            f"Market Cap: {stock_info.get('Market Cap', 'N/A')}"
        )
    
    try:
        stock_context = "\n".join([f"{k}: {v}" for k, v in stock_info.items()])
        
        news_context = "Recent News:\n" + "\n".join([
            f"- {news['title']} (Source: {news['publisher']})"
            for news in news
        ])
        
        full_context = f"{stock_context}\n\n{news_context}"
        
        # Prompt selection logic
        if query_type == "Analyst Recommendations":
            prompt = f"Analyze analyst recommendations for this stock:\n{full_context}"
        elif query_type == "Latest News Analysis":
            prompt = f"Analyze news impact on the stock:\n{full_context}"
        else:
            prompt = f"Provide comprehensive stock analysis:\n{full_context}"
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a professional financial analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Analysis Error: {e}"

# Main Streamlit App
def main():
    st.title("🚀 Advanced Financial Insight AI")
    st.markdown("Comprehensive stock analysis with AI insights")

    # Sidebar Configuration
    st.sidebar.header("🔍 Stock Analysis")
    
    # Stock Symbol Input
    stock_symbol = st.sidebar.text_input(
        "Enter Stock Symbol", 
        value="NVDA", 
        help="Enter a valid stock ticker"
    )

    # Analysis Type Selection
    query_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Comprehensive Analysis",
            "Analyst Recommendations",
            "Latest News Analysis"
        ]
    )

    # Generate Analysis Button
    if st.sidebar.button("Generate Analysis"):
        with st.spinner("Analyzing stock data..."):
            try:
                # Fetch Stock Information
                stock_info = get_stock_info(stock_symbol)
                
                if stock_info:
                    st.subheader(f"Financial Snapshot: {stock_symbol}")
                    info_df = pd.DataFrame.from_dict(stock_info, orient='index', columns=['Value'])
                    st.table(info_df)
                
                # Fetch Charts
                volatility_data = get_stock_volatility(stock_symbol)
                pe_data = get_historical_pe_ratio(stock_symbol)
                
                if volatility_data is not None:
                    st.subheader("📊 Stock Price and Volatility")
                    volatility_chart = create_price_volatility_chart(volatility_data)
                    st.plotly_chart(volatility_chart, use_container_width=True)
                
                if pe_data is not None:
                    st.subheader("📈 Historical PE Ratio")
                    pe_ratio_chart = create_pe_ratio_chart(pe_data)
                    st.plotly_chart(pe_ratio_chart, use_container_width=True)
                
                # Fetch News
                real_time_news = get_duckduckgo_news(stock_symbol)
                
                # Display News
                st.subheader("📰 Latest News")
                for news in real_time_news:
                    st.markdown(f"**{news['title']}**")
                    st.markdown(f"*Source: {news['publisher']}*")
                    st.markdown(f"[Read more]({news['link']})")
                    st.markdown("---")
                
                # Generate AI Analysis
                ai_analysis = generate_ai_analysis(stock_info, real_time_news, query_type)
                
                # Display AI Analysis
                st.subheader("🤖 AI-Powered Insights")
                st.write(ai_analysis)
                
            except Exception as e:
                st.error(f"Analysis Error: {e}")

    # Disclaimer
    st.sidebar.markdown("---")
    st.sidebar.warning(
        "🚨 Disclaimer: AI-generated analysis. "
        "Consult a financial advisor before investing."
    )

# Run the Streamlit app
if __name__ == "__main__":
    main()
