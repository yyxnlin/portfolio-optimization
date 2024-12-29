import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import warnings

"""
assumptions:
- buy/sell signals are based on the dual moving average crossover strategy, (short and long term sma lengths can be adjusted in parameters below)
- the golden cross & death cross pair will be used as the backtesting strategy
- the weights will be generated from the first time period
"""


# ****************************** parameters ******************************
tickers = ['AAPL', 'GOOG', 'MSFT'] 
start_date = '2022-12-01'
end_date = '2024-09-30'
time_period = 'M' # D = daily, M = monthly, Y = yearly
short_sma = 50
long_sma = 100
risk_free_rate = 0.01  
# **************************** DO NOT MODIFY BELOW ****************************

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

time_period_str = {'D': 'Daily', 'M': 'Monthly', 'Y': 'Yearly'}.get(time_period)
SHORT_SMA_STR = "SMA" + str(short_sma)
LONG_SMA_STR = "SMA" + str(long_sma)

tickers = sorted(tickers)
num_assets = len(tickers)

def portfolio_return(weights, req_returns):
    return np.sum(weights * req_returns.mean())

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def negative_sharpe_ratio(weights, req_returns, cov_matrix, risk_free_rate=0):
    p_return = portfolio_return(weights, req_returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility  # negative sharpe ratio for minimization

# Constraints: sum of weights must be 1
def constraint(weights):
    return np.sum(weights) - 1

def calculate_return(prices, start_date, end_date):
    start_price = prices.loc[start_date]
    end_price = prices.loc[end_date]

    if isinstance(start_price, pd.Series):
        start_price = start_price.iloc[0]
    if isinstance(end_price, pd.Series):
        end_price = end_price.iloc[0]

    return (end_price - start_price) / start_price

def calculate_max_drawdown(prices):
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[0]
    
    prices = prices.dropna()
    
    cumulative_return = prices / prices.iloc[0]
    running_max = cumulative_return.cummax()
    drawdown = cumulative_return / running_max - 1
    
    return drawdown.min()


def check_golden_death_cross(data, SHORT_SMA_STR=SHORT_SMA_STR, LONG_SMA_STR=LONG_SMA_STR):
    golden_crosses = []
    death_crosses = []
    if data[SHORT_SMA_STR].isnull().any() or data[LONG_SMA_STR].isnull().any():
        print("Warning: NaN values found in moving averages. Check initial data points.")
    
    for i in range(1, len(data)):
        if not pd.isna(data[SHORT_SMA_STR][i]) and not pd.isna(data[LONG_SMA_STR][i]):
            if data[SHORT_SMA_STR][i] >= data[LONG_SMA_STR][i] and data[SHORT_SMA_STR][i-1] < data[LONG_SMA_STR][i-1]:
                golden_crosses.append(data.index[i])
            elif data[SHORT_SMA_STR][i] <= data[LONG_SMA_STR][i] and data[SHORT_SMA_STR][i-1] > data[LONG_SMA_STR][i-1]:
                death_crosses.append(data.index[i])
    return golden_crosses, death_crosses



# **************************** PART 0: download data  **************************** 

data = yf.download(tickers, start=start_date, end=end_date)['Close']
req_data = data.resample(time_period).last()
req_returns = req_data.pct_change().dropna()
cov_matrix = req_returns.iloc[:1].cov()

# download sp500 data
sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Close']
sp500_data_normalized = sp500_data / sp500_data.iloc[0]


# **************************** PART 1: optimize weights (constant based on first period) **************************** 
initial_guess = np.ones(num_assets) / num_assets # initial guess for weights (equal allocation)
bounds = tuple((0, 1) for asset in range(num_assets))
constraints = {'type': 'eq', 'fun': constraint}

# minimize portfolio volatility to find the minimum risk portfolio
optimized_result = sco.minimize(portfolio_volatility, initial_guess, args=(cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

max_sharpe_weights = optimized_result.x

# **************************** PART 2: backtesting the strategy **************************** 
data['Portfolio'] = sum(data[ticker] * weight for ticker, weight in zip(tickers, max_sharpe_weights))
data['Portfolio_Normalized'] = data['Portfolio'] / data['Portfolio'].iloc[0]

# Calculate simple moving averages
data[SHORT_SMA_STR] = data['Portfolio'].rolling(window=short_sma, min_periods=1).mean()
data[LONG_SMA_STR] = data['Portfolio'].rolling(window=long_sma, min_periods=1).mean()


trade_strat_found = False
golden_crosses, death_crosses = check_golden_death_cross(data)

# get the first buy/sell dates
if golden_crosses and death_crosses:
    first_buy_date = golden_crosses[0]
    first_sell_date = next((date for date in death_crosses if date > first_buy_date), None)

    if first_buy_date != None and first_sell_date != None:
        trade_strat_found = True

        # calculate portfolio performance
        portfolio_return = calculate_return(data['Portfolio_Normalized'], first_buy_date, first_sell_date)
        portfolio_max_drawdown = calculate_max_drawdown(data['Portfolio_Normalized'].loc[first_buy_date:first_sell_date])

        # S&P 500 performance over the same period
        sp500_return = calculate_return(sp500_data_normalized, first_buy_date, first_sell_date)
        sp500_max_drawdown = calculate_max_drawdown(sp500_data_normalized.loc[first_buy_date:first_sell_date])

        # output results
        print("\n\n\n---------------------------------")
        print(f"Buy on {first_buy_date}")
        print(f"Sell on {first_sell_date}")
        print("---------------------------------")
        print("Portfolio Weights:")
        for ticker, weight in zip(tickers, max_sharpe_weights):
            print("{:<10} {:<.2f}%".format(ticker, weight*100))
        print("---------------------------------")
        print("{:<25} {:>15.2f}%".format("Portfolio Return", portfolio_return * 100))
        print("{:<25} {:>15.2f}%".format("Portfolio Max Drawdown", portfolio_max_drawdown * 100))
        print("{:<25} {:>15.2f}%".format("S&P 500 Return", sp500_return * 100))
        print(f"Outperformed S&P 500 by {portfolio_return-sp500_return:.2%}")

    else:
        print("No valid sell date found after the first buy date.")
else:
    print("No golden/death crosses identified. Unable to evaluate performance.")


# **************************** PART 3: plotting results **************************** 
plt.figure(figsize=(14, 7))
plt.plot(data['Portfolio_Normalized'], label='Portfolio Value', color='blue')
plt.plot(data[SHORT_SMA_STR] / data['Portfolio'].iloc[0], label=SHORT_SMA_STR+' (Portfolio)', color='gold')
plt.plot(data[LONG_SMA_STR] / data['Portfolio'].iloc[0], label=LONG_SMA_STR+' (Portfolio)', color='coral')
plt.plot(sp500_data_normalized, label='S&P 500', color='teal', linestyle='--')

if trade_strat_found:
    plt.scatter(first_buy_date, data.loc[first_buy_date, SHORT_SMA_STR]/ data['Portfolio'].iloc[0], color='green', label='Buy', marker='^', s=100)
    plt.scatter(first_sell_date, data.loc[first_sell_date, SHORT_SMA_STR]/ data['Portfolio'].iloc[0], color='red', label='Sell', marker='v', s=100)

plt.title('Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()