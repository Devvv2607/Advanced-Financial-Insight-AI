import streamlit as st
import plotly
import yfinance as yf
import pandas as pd
import plotly.graph_objs as plt
# Replaced Groq client with Gemini API key usage
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# Lad environment variables from .env file
load_dotenv()

# Streamlit App Configuration
st.set_page_config(
    page_title="Financial Analysis AI Agent",
    page_icon="💹",
    layout="wide"
)

# Initialize Gemini API key
def get_gemini_key():
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("Gemini API Key is missing. Please set GEMINI_API_KEY in your .env file.")
            return None
        return gemini_api_key
    except Exception as e:
        st.error(f"Error retrieving Gemini API key: {e}")
        return None

# Fetch Stock Information
def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        
        # Fetch key information
        info = stock.info
        
        # Create a structured dictionary of key financial metrics
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
        # Download historical stock data
        stock_data = yf.download(symbol, period=period)
        
        if stock_data.empty:
            st.error(f"No historical data found for {symbol}")
            return None
        
        # Calculate daily returns
        stock_data['Daily Returns'] = stock_data['Close'].pct_change()
        
        # Calculate rolling volatility (30-day standard deviation of returns)
        stock_data['Rolling Volatility'] = stock_data['Daily Returns'].rolling(window=30).std() * (252 ** 0.5)  # Annualized
        
        return stock_data
    except Exception as e:
        st.error(f"Error fetching historical stock data: {e}")
        return None

# Create Volatility Visualization
def create_volatility_chart(stock_data):
    try:
        # Create figure with two y-axes
        fig = plt.Figure()
        
        # Price line
        fig.add_trace(
            plt.Scatter(
                x=stock_data.index, 
                y=stock_data['Close'], 
                name='Stock Price', 
                line=dict(color='blue'),
                yaxis='y1'
            )
        )
        
        # Volatility line
        fig.add_trace(
            plt.Scatter(
                x=stock_data.index, 
                y=stock_data['Rolling Volatility'], 
                name='30-Day Volatility', 
                line=dict(color='red'),
                yaxis='y2'
            )
        )
        
        # Layout configuration
        fig.update_layout(
            title=f'Stock Price and Volatility',
            xaxis_title='Date',
            yaxis_title='Stock Price',
            yaxis2=dict(
                title='Annualized Volatility',
                overlaying='y',
                side='right'
            ),
            height=600,
            legend=dict(x=0, y=1.1, orientation='h')
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating volatility chart: {e}")
        return None

# Fetch News Using DuckDuckGo
def get_duckduckgo_news(symbol, limit=5):
    try:
        with DDGS() as ddgs:
            # Search for recent news about the stock
            news_results = list(ddgs.news(f"{symbol} stock recent news", max_results=limit))
            
            # Transform results to a consistent format
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
    api_key = get_gemini_key()
    if not api_key:
        return "Unable to generate AI analysis due to missing GEMINI_API_KEY."
    
    try:
        # Prepare context for AI
        stock_context = "\n".join([f"{k}: {v}" for k, v in stock_info.items()])
        
        # Prepare news context
        news_context = "Recent News:\n" + "\n".join([
            f"- {news['title']} (Source: {news['publisher']})"
            for news in news
        ])
        
        # Full context
        full_context = f"{stock_context}\n\n{news_context}"
        
        # Generate prompt based on query type
        if query_type == "Analyst Recommendations":
            prompt = f"Provide a comprehensive analysis of analyst recommendations for this stock. Consider the following details:\n{full_context}\n\nFocus on: current analyst ratings, price targets, and recent sentiment changes."
        elif query_type == "Latest News Analysis":
            prompt = f"Analyze the latest news and its potential impact on the stock. Consider these details:\n{full_context}\n\nProvide insights on how recent news might affect the stock's performance."
        elif query_type == "Comprehensive Analysis":
            prompt = f"Provide a holistic analysis of the stock, integrating financial metrics and recent news:\n{full_context}\n\nOffer a balanced perspective on investment potential."
        else:
            prompt = f"Generate a detailed financial and news-based analysis:\n{full_context}"
        
        # NOTE: Gemini integration placeholder
        # The API key is available in `api_key`. Replace this block with
        # actual Gemini SDK/HTTP call to generate a model response.
        placeholder_response = (
            "Gemini API key detected. Prompt prepared for model:\n\n" + prompt +
            "\n\n(Integrate Gemini SDK or REST call here to send the prompt to the Gemini model and return its output.)"
        )
        return placeholder_response
    except Exception as e:
        return f"Error generating AI analysis: {e}"

# Main Streamlit App
def main():
    st.title("🚀 Advanced Financial Insight AI")
    st.markdown("Comprehensive stock analysis with DuckDuckGo news search")

    # Sidebar Configuration
    st.sidebar.header("🔍 Stock Analysis")
    
    # Stock Symbol Input
    stock_symbol = st.sidebar.text_input(
        "Enter Stock Symbol", 
        value="NVDA", 
        help="Enter a valid stock ticker (e.g., AAPL, GOOGL)"
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
        with st.spinner("Fetching and analyzing stock data..."):
            try:
                # Fetch Stock Information
                stock_info = get_stock_info(stock_symbol)
                
                if stock_info:
                    # Display Stock Information
                    st.subheader(f"Financial Snapshot: {stock_symbol}")
                    info_df = pd.DataFrame.from_dict(stock_info, orient='index', columns=['Value'])
                    st.table(info_df)
                
                # Fetch and Display Volatility Chart
                volatility_data = get_stock_volatility(stock_symbol)
                if volatility_data is not None:
                    # Create and Display Volatility Chart
                    st.subheader("📊 Stock Price and Volatility")
                    volatility_chart = create_volatility_chart(volatility_data)
                    st.plotly_chart(volatility_chart, use_container_width=True)
                
                # Fetch News via DuckDuckGo
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
                st.error(f"An error occurred: {e}")

    # Disclaimer
    st.sidebar.markdown("---")
    st.sidebar.warning(
        "🚨 Disclaimer: This is an AI-generated analysis. "
        "Always consult with a financial advisor before making investment decisions."
    )

# Run the Streamlit app
if __name__ == "__main__":
    main()
