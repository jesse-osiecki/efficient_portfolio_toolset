#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://gist.github.com/PyDataBlog/2d5740e4199f2f898b68e154f8951ef2#file-efficient-frontier-with-quandl-part-1-py
# import needed modules
import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(101) # remove for more true randomness
num_portfolios = 500000
date_start = '2018-01-01'
selected = [
    "AMZN", # Amazon
    "XLNX", # Xilinx
    "TSLA", # Tesla
    "ADI", # Analog Devices Inc
    #"BABA",
    "BIIB", # Biogen
    "AMD", # AMD
    "SCHW", # Schwab
    "PTON", # Peloton
    "WORK", # Slack
    "SDC", # Smile Direct
    "TM", # Toyota
    "JNJ", # Johnson and Johnson
    "SLRX", # Salarius
    "LRCX", # Lam Research
    "PCG", # Pacific Gas
    "ISRG", # Intuitive Surgical
    "STAA", # Starr Surgical
]

data = yfinance.download(
        tickers = selected,
        start=date_start,
        #end='2019-06-03',
        interval = "1d",
    )

sanitized_data = data.Close[data.Close >= data.Close.quantile(0.001)] # Some yahoo data is wrong - look into it
returns_daily = sanitized_data.pct_change() # Compute the percent change with the previous element padding missing data
returns_annual = returns_daily.mean() * 252 # TODO compute avg APY correctly, this squishes the result
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 252

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
stock_weights = []
sharpe_ratio = []

# set the number of combinations for imaginary portfolios
num_assets = len(selected)

# check ourselves
if not (returns_annual.keys() == cov_annual.columns).all():
    print("Keys do not match")
    exit(1)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    # Compute Weights
    weights = np.random.random(num_assets)
    # Normalize Weights
    weights /= np.sum(weights)
    # Weighted returns
    returns = weights.dot(returns_annual)
    # Compute volatility as the weighted standard deviation
    volatility = np.sqrt(weights.T.dot(cov_annual.dot(weights)))
    # Sharpe ratio (note there isn't a risk-free asset)
    sharpe = returns / volatility
    # Construct data
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)
    sharpe_ratio.append(sharpe)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(cov_annual.columns):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in cov_annual.columns]

# reorder dataframe columns
df = df[column_order]

# find min Volatility & max sharpe values in the dataframe (df)
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

# print the details of the 2 special portfolios
print(date_start)
print("Avg Annual Returns")
print(returns_annual)
print("Min Variance Portfolio")
print(min_variance_port.T)
print("Sharpe Portfolio")
print(sharpe_portfolio.T)

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
# Optimization plot
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()
