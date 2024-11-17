import pandas as pd

# Load the CSV files into DataFrames
gold_bees = pd.read_csv('gold_bees.csv')
sensex = pd.read_csv('sensex.csv')
nifty50 = pd.read_csv('nifty50.csv')
usd_eur = pd.read_csv('usd_eur.csv')
crude_oil = pd.read_csv('crude_oil.csv')

# Perform INNER JOIN based on the 'date' column
merged_data = gold_bees.merge(sensex, on='Date', how='inner') \
    .merge(nifty50, on='Date', how='inner') \
    .merge(usd_eur, on='Date', how='inner') \
    .merge(crude_oil, on='Date', how='inner')

# Display the result
print(merged_data)

# Optionally, save the result to a new CSV file
merged_data.to_csv('merged_data.csv', index=False)
