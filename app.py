import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import talipp.indicators as ta
import openai
import os
from dotenv import load_dotenv

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
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
    data = data.head(100)  # Get the last 100 days of data
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    return data

def add_technical_indicators(data):
    """Add various technical indicators to the stock data."""
    close_prices = data['Close'].tolist()
    high_prices = data['High'].tolist()
    low_prices = data['Low'].tolist()
    volumes = data['Volume'].tolist()

    indicators = {
        'ADL': ta.ADL(),
        'Aroon': ta.Aroon(14),
        'ADX': ta.ADX(14),
        'ATR': ta.ATR(14),
        'AO': ta.AO(),
        'BOP': ta.BOP(),
        'BB': ta.BB(20, 2.0),
        'CHOP': ta.CHOP(14),
        'KST': ta.KST(),
        'MACD': ta.MACD(),
        'StochRSI': ta.StochRSI(),
        'VWAP': ta.VWAP()
    }

    for i in range(len(data)):
        indicators['ADL'].add_input_value(high_prices[i], low_prices[i], close_prices[i], volumes[i])
        indicators['Aroon'].add_input_value(high_prices[i], low_prices[i])
        indicators['ADX'].add_input_value(high_prices[i], low_prices[i], close_prices[i])
        indicators['ATR'].add_input_value(high_prices[i], low_prices[i], close_prices[i])
        indicators['AO'].add_input_value(high_prices[i], low_prices[i])
        indicators['BOP'].add_input_value(high_prices[i], low_prices[i], close_prices[i], volumes[i])
        indicators['BB'].add_input_value(close_prices[i])
        indicators['CHOP'].add_input_value(high_prices[i], low_prices[i], close_prices[i])
        indicators['KST'].add_input_value(close_prices[i])
        indicators['MACD'].add_input_value(close_prices[i])
        indicators['StochRSI'].add_input_value(close_prices[i])
        indicators['VWAP'].add_input_value(high_prices[i], low_prices[i], close_prices[i], volumes[i])

    for name, indicator in indicators.items():
        data[name] = [indicator[-1].value for _ in range(len(data))] if isinstance(indicator[-1], list) else indicator

    return data

def analyze_stock_data(data):
    """Analyze stock data using GPT-3.5 and return insights."""
    prompt = f"""
    You are a financial analyst. Analyze the following stock data and provide insights based on the technical indicators:
    {data.to_string()}
    """

    response = openai.Completion.create(
        engine="gpt-35-turbo",
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()

def plot_price_chart(data, symbol):
    """Plot the price chart for the last 20 days."""
    plt.figure(figsize=(10, 4))
    plt.plot(data['Close'].tail(20))
    plt.title(f'{symbol} Closing Prices (Last 20 Days)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    st.pyplot(plt)

def main():
    """Main function to run the Streamlit app."""
    st.title('Stock Analysis with Technical Indicators')

    symbol = st.text_input('Enter US listed stock symbol (e.g., IBM):', 'IBM')

    if st.button('Analyze'):
        with st.spinner('Fetching data...'):
            stock_data = fetch_stock_data(symbol)

        with st.spinner('Calculating technical indicators...'):
            stock_data = add_technical_indicators(stock_data)

        st.subheader(f'{symbol} Stock Data and Technical Indicators')
        st.dataframe(stock_data)

        st.subheader(f'{symbol} Price Chart (Last 20 Days)')
        plot_price_chart(stock_data, symbol)

        with st.spinner('Analyzing with GPT-3.5...'):
            insights = analyze_stock_data(stock_data)

        st.subheader('Insights from GPT-3.5')
        st.write(insights)

if __name__ == "__main__":
    main()
