import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import warnings



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



# **************************** PART 3: plotting results **************************** 
plt.figure(figsize=(14, 7))
plt.plot(data['Portfolio_Normalized'], label='Portfolio Value', color='blue')
plt.plot(sp500_data_normalized, label='S&P 500', color='teal', linestyle='--')

plt.title('Optimal (Tangent) Portfolio Performance vs. S&P500')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()