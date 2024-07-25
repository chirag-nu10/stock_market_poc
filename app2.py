import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import talipp.indicators as ta
from talipp.ohlcv import OHLCV
import openai
import regex as re
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Alpha Vantage and OpenAI API credentials
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
openai.api_type = "azure"
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')

def fetch_stock_data(symbol):
    """Fetch daily stock data for the past 100 days from Alpha Vantage."""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)

    data = response.json()
    data = pd.DataFrame(data['Time Series (Daily)']).T

    # data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
    #data = data.head(100)  # Get the last 100 days of data
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    data = data.sort_index()
    
    # data = pd.read_csv("IBM Historical Data.csv")
    # data = data.rename(columns={
    #     'Open': 'Open',
    #     'High': 'High',
    #     'Low': 'Low',
    #     'Price': 'Close',
    #     'Vol.': 'Volume'
    # })
    
    print("###########################")
    print(data.dtypes)
    return data

# def add_technical_indicators(data):
#     """Add various technical indicators to the stock data."""
#     close_prices = data['Close'].tolist()
#     high_prices = data['High'].tolist()
#     low_prices = data['Low'].tolist()
#     volumes = data['Volume'].tolist()

#     indicators = {
#         'Aroon': ta.Aroon(14),
#         'ADX': ta.ADX(14,14),
#         'ATR': ta.ATR(14),
#         'AO': ta.AO(9,14),
#         'BB': ta.BB(20, 2.0),
#         'CHOP': ta.CHOP(14),
#         #'KST': ta.KST(10, 10, 15, 15, 20, 20, 30, 30, 9),
#         'MACD': ta.MACD(12, 26, 9),
#         'StochRSI': ta.StochRSI(14,3,3,3),
#     }

#     print(indicators)

#     for i in range(len(data)):
#         print("############################    i VALUE    ##################",i)
#         indicators['Aroon'].add_input_value(OHLCV(high_prices[i], low_prices[i], close_prices[i], volumes[i]))
#         indicators['ADX'].add_input_value(OHLCV(high_prices[i], low_prices[i], close_prices[i], volumes[i]))
#         indicators['ATR'].add_input_value(OHLCV(high_prices[i], low_prices[i], close_prices[i], volumes[i]))
#         indicators['AO'].add_input_value(OHLCV(high_prices[i], low_prices[i], close_prices[i], volumes[i]))
#         indicators['BB'].add_input_value(OHLCV(high_prices[i], low_prices[i], close_prices[i], volumes[i]))
#         indicators['CHOP'].add_input_value(OHLCV(high_prices[i], low_prices[i], close_prices[i], volumes[i]))
#         #indicators['KST'].add_input_value(OHLCV(high_prices[i], low_prices[i], close_prices[i], volumes[i]))
#         indicators['MACD'].add_input_value(OHLCV(high_prices[i], low_prices[i], close_prices[i], volumes[i]))
#         indicators['StochRSI'].add_input_value(OHLCV(high_prices[i], low_prices[i], close_prices[i], volumes[i]))

#     for name, indicator in indicators.items():
#         data[name] = [indicator[-1].value for _ in range(len(data))] if isinstance(indicator[-1], list) else indicator

#     return data

def add_technical_indicators(data):
    """Add various technical indicators to the stock data."""
    # close_prices = data['Close'].tolist()
    # high_prices = data['High'].tolist()
    # low_prices = data['Low'].tolist()
    # open_prices = data['Open'].tolist()
    # volumes = data['Volume'].tolist()

    close_prices = [float(price) for price in data['Close'].tolist()]
    high_prices = [float(price) for price in data['High'].tolist()]
    low_prices = [float(price) for price in data['Low'].tolist()]
    open_prices = [float(price) for price in data['Open'].tolist()]
    volumes = [float(volume) for volume in data['Volume'].tolist()]

    indicators = {
        'Aroon': ta.Aroon(14),
        'ADX': ta.ADX(14, 14),
        'ATR': ta.ATR(14),
        'AO': ta.AO(9, 14),
        'BB': ta.BB(20, 2.0),
        'CHOP': ta.CHOP(14),
        'KST': ta.KST(10, 10, 15, 15, 20, 20, 30, 30, 9),
        'MACD': ta.MACD(12, 26, 9),
        'StochRSI': ta.StochRSI(14, 3, 3, 3),
    }

    # Add OHLCV values for indicators that need full OHLCV data
    for i in range(len(data)):
        ohlcv = OHLCV(open_prices[i], high_prices[i], low_prices[i], close_prices[i], volumes[i])
        indicators['Aroon'].add(ohlcv)
        indicators['ADX'].add(ohlcv)
        indicators['ATR'].add(ohlcv)
        indicators['AO'].add(ohlcv)
        indicators['CHOP'].add(ohlcv)

    # Add close prices for indicators that need only the close price
    for i in range(len(close_prices)):
        indicators['BB'].add(close_prices[i])
        indicators['KST'].add(close_prices[i])
        indicators['MACD'].add(close_prices[i])
        indicators['StochRSI'].add(close_prices[i])

    # Extract indicator values and add to the DataFrame
    for name, indicator in indicators.items():
        if name in ['BB', 'KST', 'MACD', 'StochRSI']:
            # Add indicator values for close price-based indicators
            data[name] = [indicator[i] for i in range(len(close_prices))]
        else:
            # Add indicator values for OHLCV-based indicators
            data[name] = [indicator[i] for i in range(len(data))]

    return data


def analyze_stock_data(data):
    """Analyze stock data using GPT-3.5 and return insights."""
    prompt = f"""
    You are a financial analyst. Analyze the following stock data and provide insights based on the technical indicators:
    {data.to_string()}
    """
    response = openai.chat.completions.create(
        # model="sermo",
        model = "nu10",
        messages=[
                {"role": "system", "content": "You are a financial analyst, helping user to analyze stock based on technical indicators."},
                {"role": "user", "content": f"Analyze the following stock data and provide detailed insights in the following format:\n1. Conclusion & course of action for investor \n2. List down the reason to reach above conclusion in bullet points.\n Data:\n'''{data.to_string()}'''"}
            ],
        max_tokens=1000
    )

    return response.choices[0].message.content

def plot_price_chart(data, symbol):
    """Plot the price chart for the last 20 days."""
    plt.figure(figsize=(10, 4))
    plt.plot(data['Close'].tail(20).astype(float))
    plt.title(f'{symbol} Closing Prices (Last 20 Days)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45) 
    st.pyplot(plt)

def gpt_genie_says(insights):
    # This function determines the final recommendation (BUY, HOLD, SELL)
    response = openai.chat.completions.create(
        model="nu10",
        messages=[
                {"role": "system", "content": "You are a helper, user send you insight about stock and you have to classify it in BUY, SELL or, HOLD"},
                {"role": "user", "content": f"Based on the provided stock data, I have analyzed the technical indicators and reached the following conclusion and course of action for the investor:1. Conclusion & course of action for investor: - SELL/AVOID further investment in the stock 2. Reasons for the above conclusion:- Multiple technical indicators are showing bearish signals, indicating a potential downward trend in the stock price. - Aroon indicator consistently showing a strong downward trend. - ADX values indicate a strengthening downtrend.- Stochastic RSI values have been consistently low, indicating the stock is oversold.- MACD values are consistently negative, indicating a bearish trend. - KST values consistently below the signal line, indicating a bearish trend.- The stock price is consistently below the lower Bollinger Band, indicating a potential oversold condition.Based on the above analysis, it is advisable for the investor to consider selling the stock or avoiding further investment until bullish signals are observed in the technical indicators."},
                {"role": "assistant", "content": "SELL"},
                {"role": "user", "content": f"'''{insights}'''"}
            ],
        max_tokens=500
    )

    return response.choices[0].message.content

def main():
    """Main function to run the Streamlit app."""
    st.title('Stock Analysis with Technical Indicators')

    symbol = st.text_input('Enter US listed stock symbol (e.g., IBM):', 'IBM')

    if st.button('Analyze'):

        try:
            with st.spinner('Fetching data...'):
                stock_data = fetch_stock_data(symbol)
                st.success('Data fetched successfully')
        except Exception as e:
            st.error(f'Error fetching data: {e}')
            return

        # st.write(stock_data.tail())

        try:
            with st.spinner('Calculating technical indicators...'):
                stock_data = add_technical_indicators(stock_data)
                st.success('Technical indicators calculated successfully')
        except Exception as e:
            st.error(f'Error calculating technical indicators: {e}')
            return

        try:
            with st.spinner('Analyzing with GPT-3.5...'):
                insights = analyze_stock_data(stock_data.tail(20))
                st.success('Analysis completed')
        except Exception as e:
            st.error(f'Error analyzing data with GPT-3.5: {e}')
            return

        st.subheader('Insights from GPT')
        st.write(insights)

        st.subheader(f'{symbol} Price Chart (Last 20 Days)')
        plot_price_chart(stock_data, symbol)

        st.subheader(f'{symbol} Stock Data and Technical Indicators')
        st.dataframe(stock_data)       

def main2():
    data = fetch_stock_data("IBM")
    data2 = add_technical_indicators(data)
    print(data2.tail())
    response_gpt = analyze_stock_data(data2.tail(20))
    print(response_gpt)

def main3():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Stock Analysis App",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title('📈 Stock Analysis with Technical Indicators')
    
    with st.sidebar:
        st.image("nu10-logo.png", width=75)
        st.header("Stock Symbol Input")
        symbol = st.text_input('Enter US listed stock symbol (e.g., IBM):', 'IBM')
        analyze_button = st.button('Analyze')
    
    if analyze_button:
        st.markdown("---")
        
        try:
            with st.spinner('Fetching data...'):
                stock_data = fetch_stock_data(symbol)
                st.success('Data fetched successfully')
        except Exception as e:
            st.error(f'Error fetching data: {e}')
            return

        try:
            with st.spinner('Calculating technical indicators...'):
                stock_data = add_technical_indicators(stock_data)
                st.success('Technical indicators calculated successfully')
        except Exception as e:
            st.error(f'Error calculating technical indicators: {e}')
            return

        try:
            with st.spinner('Analyzing with GPT-3.5...'):
                insights = analyze_stock_data(stock_data.tail(20))
                st.success('Analysis completed')
        except Exception as e:
            st.error(f'Error analyzing data with GPT-3.5: {e}')
            return
        
        st.subheader('📊 Insights from GPT')
        st.write(insights)
        
        st.markdown("---")
        st.subheader(f'📈 {symbol} Price Chart (Last 20 Days)')
        plot_price_chart(stock_data, symbol)
        
        st.markdown("---")
        st.subheader("🔍 Detailed Data")
        st.dataframe(stock_data)

def main4():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Stock Analysis App",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title('📈 Stock Analysis with Technical Indicators')
    
    with st.sidebar:
        st.image("nu10-logo.png", width=75)
        st.header("Stock Symbol Input")
        symbol = st.text_input('Enter US listed stock symbol (e.g., IBM):', 'IBM')
        analyze_button = st.button('Analyze')
    
    if analyze_button:
        st.markdown("---")
        
        try:
            with st.spinner('Fetching data...'):
                stock_data = fetch_stock_data(symbol)
                st.success('Data fetched successfully')
        except Exception as e:
            st.error(f'Error fetching data: {e}')
            return

        try:
            with st.spinner('Calculating technical indicators...'):
                stock_data = add_technical_indicators(stock_data)
                st.success('Technical indicators calculated successfully')
        except Exception as e:
            st.error(f'Error calculating technical indicators: {e}')
            return

        try:
            with st.spinner('Analyzing with GPT-3.5...'):
                insights = analyze_stock_data(stock_data.tail(20))
                st.success('Analysis completed')
        except Exception as e:
            st.error(f'Error analyzing data with GPT-3.5: {e}')
            return
        
        print(insights)

        st.subheader('📊 Insights from GPT')

        # Use regex to split the insights into sections with more flexible patterns
        conclusion_patterns = [
            r"1.(.*?)2",
        ]
        
        reasons_patterns = [
            r"2.(.*)",
        ]
        
        conclusion = None
        reasons = None
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, insights, re.DOTALL | re.IGNORECASE)
            if match:
                conclusion = match.group(1).strip()
                break
        
        for pattern in reasons_patterns:
            match = re.search(pattern, insights, re.DOTALL | re.IGNORECASE)
            if match:
                reasons = match.group(1).strip()
                break
        
        if not conclusion:
            conclusion = "Conclusion not found."
        if not reasons:
            reasons = "Reasons not found."

        # Display conclusion and reasons in separate expanders
        with st.expander("Conclusion & Course of Action for Investor"):
            st.write(conclusion)
        with st.expander("Reasons for the Above Conclusion"):
            st.write(reasons)
        
        st.markdown("---")
        st.subheader(f'📈 {symbol} Price Chart (Last 20 Days)')
        plot_price_chart(stock_data, symbol)
        
        st.markdown("---")
        st.subheader("🔍 Detailed Data")
        st.dataframe(stock_data)

def main5():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Stock Analysis App",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.image("gptgenie.png", width=250)
    # st.title('GPT Genie')

    # col1, col2 = st.columns([1, 12])
    # with col1:
    #     st.image("ginie.png", width=60)
    # with col2:
    #     st.title('GPT Genie')
    
    with st.sidebar:
        st.image("nu10-logo.png", width=75)
        st.header("Stock Symbol Input")

        symbols = eval(f"[{st.text_input('Enter US listed stock symbols (e.g., IBM,GOOGL,TSLA):')}]")
        percentage = eval(f"[{st.text_input('Enter percentages in sequence of companies (Total must be 100) (e.g., 30,40,30):')}]")

        analyze_button = st.button('Analyze')
        
    if analyze_button:
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        try:
            with st.spinner('Fetching data...'):
                stock_data = fetch_stock_data(symbol)
                with col1:
                    st.success('Data fetched successfully')
        except Exception as e:
            st.error(f'Error fetching data: {e}')
            return

        try:
            with st.spinner('Calculating technical indicators...'):
                stock_data = add_technical_indicators(stock_data)
                with col2:
                    st.success('Technical indicators calculated successfully')
        except Exception as e:
            st.error(f'Error calculating technical indicators: {e}')
            return

        try:
            with st.spinner('Analyzing with GPT-3.5...'):
                insights = analyze_stock_data(stock_data.tail(20))
                print(f'Insights of OpenAI : \n\n {insights}')
                recommendation_0 = gpt_genie_says(insights)
                pattern_rec = r'\b(BUY|SELL|HOLD)\b'
                matches = re.search(pattern_rec, recommendation_0)
                recommendation = matches.group(1)
                with col3:
                    st.success('Analysis completed')
        except Exception as e:
            st.error(f'Error analyzing data with GPT-3.5: {e}')
            return
        
        
        color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}[recommendation]

        st.markdown("")

        if recommendation == "BUY":
            st.image("Buy.png", width=300)
        elif recommendation == "SELL":
            st.image("Sell.png", width=300)
        else:
            st.image("Hold.png", width=300)
        
        # st.markdown(f"### 🧞 GPT Genie says: <span style='color:{color}'>{recommendation}</span>", unsafe_allow_html=True)
        # st.markdown(f"""
        #             <div style="display: flex; align-items: flex-start;">
        #             <img src="data:image/png;base64,{st.image('ginie.png', use_column_width=True).data}" width="10">
        #             <h3 style="margin-left: 10px;"> GPT Genie says: <span style='color:{color}'>{recommendation}</span></h3>
        #             </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader('📊 Insights from GPT')

        # Use regex to split the insights into sections with more flexible patterns
        conclusion_patterns = [
            r"1\.(.*?)2",
            r"Conclusion\.(.*?)Reasons",
        ]
        
        reasons_patterns = [
            r"2\.(.*)",
            r"Reasons\.(.*)",
        ]
        
        conclusion = None
        reasons = None
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, insights, re.DOTALL | re.IGNORECASE)
            if match:
                conclusion = match.group(1).strip()
                break
        
        for pattern in reasons_patterns:
            match = re.search(pattern, insights, re.DOTALL | re.IGNORECASE)
            if match:
                reasons = match.group(1).strip()
                break
        
        if not conclusion:
            conclusion = "Conclusion not found."
        if not reasons:
            reasons = "Reasons not found."

        # Display conclusion and reasons in separate expanders
        with st.expander("Conclusion & Course of Action for Investor"):
            st.write(conclusion)
        with st.expander("Reasons for the Above Conclusion"):
            st.write(reasons)
        
        st.markdown("---")
        st.subheader(f'📈 {symbol} Price Chart (Last 20 Days)')
        plot_price_chart(stock_data, symbol)
        
        st.markdown("---")
        st.subheader("🔍 Detailed Data")
        st.dataframe(stock_data)

if __name__ == "__main__":
    main5()


# Invest smart with GPT Genie
# Indicator used name
# BUY | Hold | SELL