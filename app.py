from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import yfinance as yf
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = joblib.load('best_gold_price_prediction.pkl')
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

app = Flask(__name__)

def fetch_single_ticker(symbol):
    """Fetch data for a single ticker with error handling."""
    try:
        logger.info(f"Fetching data for {symbol}")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            prev_close = data['Open'].iloc[0]
            price_change = ((current_price - prev_close) / prev_close) * 100
            return {
                'price': round(float(current_price), 2),
                'change': round(float(price_change), 2)
            }
        else:
            logger.warning(f"No data returned for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
    return {'price': 0, 'change': 0}

def fetch_market_data():
    """Fetch market data concurrently."""
    symbols = {
        'Nifty50': '^NSEI',
        'Sensex': '^BSESN',
        'EUR/USD': 'EURUSD=X',
        'Gold Bees': 'GOLDBEES.NS',
        'Crude Oil (USD)': 'CL=F'
    }

    market_data = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(fetch_single_ticker, symbol): market
            for market, symbol in symbols.items()
        }
        for future in as_completed(future_to_symbol):
            market = future_to_symbol[future]
            try:
                data = future.result()
                market_data[market] = data
            except Exception as e:
                logger.error(f"Error processing {market}: {e}")
                market_data[market] = {'price': 0, 'change': 0}

    return market_data

@app.route('/')
def home():
    try:
        market_data = fetch_market_data()
        return render_template('index.html', market_data=market_data)
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return "Error loading page", 500

@app.route('/api/market-data')
def get_market_data():
    try:
        data = fetch_market_data()
        response = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in get_market_data: {e}")
        return jsonify({'error': 'Failed to fetch market data'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            features = [
                float(data['sensex']),
                float(data['nifty50']),
                float(data['crude_oil_usd']),
                float(data['eur_usd'])
            ]
            prediction = model.predict([features])
            return jsonify({'prediction': round(float(prediction[0]), 2)})
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, render_template, request, jsonify
# import joblib
# import numpy as np
# import yfinance as yf
# from datetime import datetime
# import logging
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Load the trained model
# try:
#     model = joblib.load('best_gold_price_prediction.pkl')
# except Exception as e:
#     logger.error(f"Error loading model: {e}")
#     raise
#
# app = Flask(__name__)
#
# def fetch_single_ticker(symbol):
#     """Fetch data for a single ticker with error handling."""
#     try:
#         logger.info(f"Fetching data for {symbol}")
#         ticker = yf.Ticker(symbol)
#         data = ticker.history(period='1d')
#         if not data.empty:
#             current_price = data['Close'].iloc[-1]
#             prev_close = data['Open'].iloc[0]
#             price_change = ((current_price - prev_close) / prev_close) * 100
#             return {
#                 'price': round(float(current_price), 2),
#                 'change': round(float(price_change), 2)
#             }
#         else:
#             logger.warning(f"No data returned for {symbol}")
#     except Exception as e:
#         logger.error(f"Error fetching {symbol}: {e}")
#     return {'price': 0, 'change': 0}
#
# def fetch_market_data():
#     """Fetch market data concurrently."""
#     symbols = {
#         'sensex': '^BSESN',
#         'nifty50': '^NSEI',
#         'eurusd': 'EURUSD=X',
#         'goldbees': 'GOLDBEES.NS',
#         'crude_oil': 'CL=F'
#     }
#
#     market_data = {}
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         future_to_symbol = {
#             executor.submit(fetch_single_ticker, symbol): market
#             for market, symbol in symbols.items()
#         }
#         for future in as_completed(future_to_symbol):
#             market = future_to_symbol[future]
#             try:
#                 data = future.result()
#                 market_data[market] = data
#             except Exception as e:
#                 logger.error(f"Error processing {market}: {e}")
#                 market_data[market] = {'price': 0, 'change': 0}
#
#     return market_data
#
# @app.route('/')
# def home():
#     try:
#         market_data = fetch_market_data()
#         return render_template('index.html', market_data=market_data)
#     except Exception as e:
#         logger.error(f"Error rendering template: {e}")
#         return "Error loading page", 500
#
# @app.route('/api/market-data')
# def get_market_data():
#     try:
#         data = fetch_market_data()
#         response = {
#             'data': data,
#             'timestamp': datetime.now().isoformat()
#         }
#         return jsonify(response)
#     except Exception as e:
#         logger.error(f"Error in get_market_data: {e}")
#         return jsonify({'error': 'Failed to fetch market data'}), 500
#
# @app.route('/api/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             data = request.get_json()
#             features = [
#                 float(data['sensex']),
#                 float(data['nifty50']),
#                 float(data['crude_oil_usd']),
#                 float(data['eur_usd'])
#             ]
#             prediction = model.predict([features])
#             return jsonify({'prediction': round(float(prediction[0]), 2)})
#         except Exception as e:
#             logger.error(f"Error in predict: {e}")
#             return jsonify({'error': 'Prediction failed'}), 500
#
# if __name__ == '__main__':
#     app.run(debug=True)