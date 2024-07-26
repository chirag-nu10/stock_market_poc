import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import talipp.indicators as ta
from talipp.ohlcv import OHLCV
import openai
import regex as re
import os
from dotenv import load_dotenv
import requests
import json
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
openai.api_type = "azure"
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')

def fetchStockData(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    data = pd.DataFrame(data['Time Series (Daily)']).T
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    data = data.sort_index()
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric)
    return data

def get_weighted_data(pairs):
    num_stocks = len(pairs)
    total_quantity = 0
    for i in range(num_stocks):
        total_quantity += pairs[i][1]
        if i == 0:
            data = fetchStockData(pairs[i][0]) * pairs[i][1]
        else:
            data += fetchStockData(pairs[i][0]) * pairs[i][1]
    data['Volume'] = data['Volume'] / total_quantity
    return data

def add_technical_indicators(data):
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

    for i in range(len(data)):
        ohlcv = OHLCV(open_prices[i], high_prices[i], low_prices[i], close_prices[i], volumes[i])
        indicators['Aroon'].add(ohlcv)
        indicators['ADX'].add(ohlcv)
        indicators['ATR'].add(ohlcv)
        indicators['AO'].add(ohlcv)
        indicators['CHOP'].add(ohlcv)

    for i in range(len(close_prices)):
        indicators['BB'].add(close_prices[i])
        indicators['KST'].add(close_prices[i])
        indicators['MACD'].add(close_prices[i])
        indicators['StochRSI'].add(close_prices[i])

    for name, indicator in indicators.items():
        if name in ['BB', 'KST', 'MACD', 'StochRSI']:
            data[name] = [indicator[i] for i in range(len(close_prices))]
        else:
            data[name] = [indicator[i] for i in range(len(data))]

    return data

def analyze_stock_data(data):
    prompt = f"""
    You are a financial analyst. Analyze the following stock data and provide insights based on the technical indicators:
    {data.to_string()}
    """
    response = openai.ChatCompletion.create(
        model="nu10",
        messages=[
            {"role": "system", "content": "You are a financial analyst, helping user to analyze stock based on technical indicators."},
            {"role": "user", "content": f"Analyze the following stock data and provide detailed insights in the following format:\n1. Conclusion & course of action for investor \n2. List down the reason to reach above conclusion in bullet points.\n Data:\n'''{data.to_string()}'''"}
        ],
        max_tokens=1000
    )
    return response.choices[0].message['content']

def gpt_genie_says(insights):
    response = openai.ChatCompletion.create(
        model="nu10",
        messages=[
            {"role": "system", "content": "You are a helper, user send you insight about stock and you have to classify it in BUY, SELL or, HOLD"},
            {"role": "user", "content": f"'''{insights}'''"}
        ],
        max_tokens=500
    )
    return response.choices[0].message['content']

@app.route('/analyzestocks', methods=['POST'])
def analyze_stocks():
    try:
        data = request.get_json()
        stock = data['stock']
        stock_data = fetchStockData(stock)
        stock_data = add_technical_indicators(stock_data)
        insights = analyze_stock_data(stock_data.tail(20))
        recommendation = gpt_genie_says(insights)

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

        result = {
            'recommendation': recommendation,
            'conclusion': conclusion,
            'reasons': reasons
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyzeportfolio', methods=['POST'])
def analyze_portfolio():
    try:
        data = request.get_json()
        pairs = data['pairs']  # list of tuples [(stock1, quantity1), (stock2, quantity2), ...]
        stock_data = get_weighted_data(pairs)
        stock_data = add_technical_indicators(stock_data)
        insights = analyze_stock_data(stock_data.tail(20))
        recommendation = gpt_genie_says(insights)

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

        result = {
            'recommendation': recommendation,
            'conclusion': conclusion,
            'reasons': reasons
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
