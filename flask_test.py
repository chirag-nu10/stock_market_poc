import requests

url = 'http://127.0.0.1:5000/analyzestocks'
headers = {'Content-Type': 'application/json'}
data = {
    "stock": "AAPL"
}

response = requests.post(url, headers=headers, json=data)
print("xxxxxxxxxxxxxxxxx Stock Analysis xxxxxxxxxxxxxxxxx")
print(response.json())

url = 'http://127.0.0.1:5000/analyzeportfolio'
headers = {'Content-Type': 'application/json'}
data = {
    "portfolio": [
        {"stock": "AAPL", "quantity": 10},
        {"stock": "MSFT", "quantity": 15}
    ]
}

response = requests.post(url, headers=headers, json=data)
print("xxxxxxxxxxxxxxxxx Portfolio Analysis xxxxxxxxxxxxxxxxx")
print(response.json())