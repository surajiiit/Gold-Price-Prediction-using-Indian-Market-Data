import yfinance as yf

# Define the ticker symbol
ticker_symbol = "^BSESN"

# Download data for Crude Oil Dec 24
data = yf.download(ticker_symbol, start="2009-01-02", end="2023-12-29")

# Save to CSV
data.to_csv("sensex.csv")

print("Data downloaded and saved as 'sensex.csv'")
