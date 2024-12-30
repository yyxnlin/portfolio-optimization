import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco


"""
assumptions:
- no short selling (all weights <= 1)
"""

# ****************************** parameters ******************************
start_date = "2021-01-01"
end_date = "2024-11-30"
time_period = 'M' # D = daily, M = monthly, Y = yearly
tickers = ['AAPL', 'GOOG', 'MSFT', 'MCD']
risk_free_rate = 0


# **************************** DO NOT MODIFY BELOW ****************************
time_period_str = {'D': 'Daily', 'M': 'Monthly', 'Y': 'Yearly'}.get(time_period)

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

def max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    # Calculate the running maximum
    running_max = cumulative_returns.cummax()

    # Calculate drawdown (current cumulative return vs. the historical maximum)
    drawdown = (cumulative_returns - running_max) / running_max

    # Find the maximum drawdown (the lowest value in the drawdown series)
    return drawdown.min()


# **************************** PART 0: download data and plot graphs  **************************** 
data = yf.download(tickers, start_date, end_date)['Close']
req_data = data.resample(time_period).last()
req_returns = req_data.pct_change().dropna()
cov_matrix = req_returns.cov()
corr_matrix = req_returns.corr()
mean_returns = req_returns.mean()
volatilities = req_returns.std()

# 1. output matrices
print("\nCorrelation Matrix:")
print(corr_matrix)

print("\n\nCovariance Matrix:")
print(cov_matrix)

print(f"\n\nExpected {time_period_str} Returns for each stock:\n{mean_returns}")


# 2. plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# subplot 1: closing prices
axes[0].set_title('Closing Prices of Selected Stocks (2021-2024)', fontsize=16)
for ticker in tickers:
    axes[0].plot(data.index, data[ticker], label=ticker)
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Closing Price (USD)', fontsize=12)
axes[0].legend(title="Tickers", fontsize=10)
axes[0].tick_params(axis='x', rotation=45)

# subplot 2: returns
axes[1].set_title(f'{time_period_str} Returns (2021-2024)', fontsize=16)
for ticker in req_returns.columns:
    axes[1].plot(req_returns.index, req_returns[ticker], label=ticker)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel(f'{time_period_str} Return', fontsize=12)
axes[1].legend()

# subplot 3: covariance matrix
sns.heatmap(cov_matrix, annot=False, cmap='twilight_shifted', fmt='.2f', linewidths=0.5, ax=axes[2])
axes[2].set_title(f'Covariance Matrix of {time_period_str} Stock Returns (2021-2024)', fontsize=16)
axes[2].set_xlabel('Assets', fontsize=12)
axes[2].set_ylabel('Assets', fontsize=12)

plt.tight_layout()
plt.show()



# **************************** PART 1: equal allocation profile **************************** 
equal_weights = np.ones(num_assets) / num_assets  # Equal weight for each asset
portfolio_return_equal_alloc = portfolio_return(equal_weights, req_returns)
portfolio_volatility_equal_alloc = portfolio_volatility(equal_weights, cov_matrix)


# **************************** PART 2: minimum risk profile **************************** 
initial_guess = np.ones(num_assets) / num_assets # initial guess for weights (equal allocation)
bounds = tuple((0, 1) for asset in range(num_assets))
constraints = {'type': 'eq', 'fun': constraint}

# minimize portfolio volatility to find the minimum risk portfolio
optimal_result = minimize(portfolio_volatility, initial_guess, args=(cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

min_risk_weights = optimal_result.x
min_risk_return = np.dot(min_risk_weights, mean_returns)
min_risk_volatility = np.sqrt(np.dot(min_risk_weights.T, np.dot(cov_matrix, min_risk_weights)))



# **************************** PART 3: random weights for all portfolio possibilities  **************************** 
# generate random portfolio weights that sum to 1
weights = np.random.random((10000, num_assets))
weights = weights / np.sum(weights, axis=1)[:, None]  # Normalize to sum to 1

# calculate portfolio returns and volatilities
portfolio_returns = np.dot(weights, mean_returns)  # Expected portfolio returns
portfolio_volatilities = np.sqrt(np.sum(weights * (weights @ cov_matrix), axis=1))  # Portfolio volatilities

# calculate sharpe ratio
excess_returns = portfolio_returns - risk_free_rate
sharpe_ratios = excess_returns / portfolio_volatilities



# **************************** PART 4: find max sharpe ratio  **************************** 
bounds = tuple((0, 1) for _ in range(len(tickers)))

# initial guess (evenly distributed portfolio)
initial_guess = [1. / len(tickers)] * len(tickers)

# optimization to find the weights that maximize sharpe ratio
optimized_result = sco.minimize(negative_sharpe_ratio, initial_guess, args=(req_returns, cov_matrix), 
                                method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': constraint})

max_sharpe_weights = optimized_result.x

# calculate return, volatility, and sharpe ratio for the optimized portfolio
max_sharpe_return = portfolio_return(max_sharpe_weights, req_returns)
max_sharpe_volatility = portfolio_volatility(max_sharpe_weights, cov_matrix)
max_sharpe_ratio = -(optimized_result.fun)  # Since we minimized the negative Sharpe ratio

# max drawdown for this portfolio
monthly_portfolio_returns = (req_returns * max_sharpe_weights).sum(axis=1)
portfolio_monthly_max_drawdown = max_drawdown(monthly_portfolio_returns)


# find the corresponding CML
# formula: y = risk_free_rate + (opt_return - risk_free_rate) / opt_volatility * x
x = np.linspace(min_risk_volatility*0.9, np.max(portfolio_volatilities)*0.9, 1000)
opt_cml = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_volatility * x



# display the results
print("\n\n\n-------------------------------")
print("Maximum Sharpe Ratio Portfolio:")
print("-------------------------------")
print("{:<25} {:.2f}".format("Sharpe Ratio", max_sharpe_ratio))
print("{:<25} {:.2f}%".format("Expected Annual Return", max_sharpe_return*100))
print("{:<25} {:.2f}%".format("Annual Volatility", max_sharpe_volatility*100))
print("{:<25} {:.2f}%".format(f"Max drawdown ({time_period_str})", portfolio_monthly_max_drawdown * 100))
print("--------------------")
print("Portfolio Weights:")
print("--------------------")

# print each ticker and its corresponding weight
for ticker, weight in zip(tickers, max_sharpe_weights):
    print("{:<10} {:<.2f}%".format(ticker, weight*100))
print("-------------------------------")





# **************************** PART 5: get efficient frontier  **************************** 
volatility_min = portfolio_volatilities.min()
volatility_max = portfolio_volatilities.max()
volatility_range = volatility_max - volatility_min

threshold = volatility_range/1000

efficient_frontier = []
unique_volatilities = np.unique(portfolio_volatilities)  

for vol in unique_volatilities:
    close_indices = np.where(portfolio_volatilities - vol < threshold)[0] 
    if close_indices.size > 0: 
        max_return = portfolio_returns[close_indices].max()
        efficient_frontier.append((vol, max_return))

efficient_frontier = np.array(efficient_frontier)





# **************************** PART 6: plot results  **************************** 
plt.figure(figsize=(10, 6))

# plotting the portfolios with random weights
plt.scatter(portfolio_volatilities, portfolio_returns, c=sharpe_ratios, cmap='YlGnBu', marker='o', s=10)
plt.colorbar(label="Sharpe Ratio")

plt.plot(efficient_frontier[:, 0], efficient_frontier[:, 1], color='deepskyblue', linewidth=2, label="Efficient Frontier")
plt.grid(True)

# single stock, equal allocations, min risk, max sharpe ratio, and CML
plt.scatter(volatilities, mean_returns, color='blue', marker='o', s=40)
plt.scatter(portfolio_volatility_equal_alloc, portfolio_return_equal_alloc, color='red', marker='o', s=40, label='Equal Allocation Portfolio')
plt.scatter(min_risk_volatility, min_risk_return, color='green', marker='D', s=40, label='Minimum Risk Portfolio')
plt.scatter(max_sharpe_volatility, max_sharpe_return, color='gold', marker='*', s=200, label="Max Sharpe Ratio Portfolio")
plt.plot(x, opt_cml, label="CML", color='orange', linestyle='--')

# labels and title
for i, ticker in enumerate(tickers):
    plt.text(volatilities[i], mean_returns[i], ticker, fontsize=12, ha='right')

plt.title('Expected Return vs. Standard Deviation (Volatility) for Assets')
plt.xlabel('Standard Deviation (Volatility)')
plt.ylabel('Expected Return')
plt.legend()
plt.show()


# **************************** PART 7: backtesting the strategy & compare with s&p500 **************************** 

# download sp500 data
sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Close']
sp500_data_normalized = sp500_data / sp500_data.iloc[0]


data['Portfolio'] = sum(data[ticker] * weight for ticker, weight in zip(tickers, max_sharpe_weights))
data['Portfolio_Normalized'] = data['Portfolio'] / data['Portfolio'].iloc[0]

outperformance = (data['Portfolio_Normalized'].iloc[-1] - sp500_data_normalized.iloc[-1]).iloc[0]

print ("Total return difference in comparison to S&P 500: {:<.2f}%".format(outperformance*100))

# **************************** PART 8: plotting results **************************** 
plt.figure(figsize=(14, 7))
plt.plot(data['Portfolio_Normalized'], label='Portfolio Value', color='skyblue')
plt.plot(sp500_data_normalized, label='S&P 500', color='midnightblue', linestyle='--')

plt.title('Optimal (Tangent) Portfolio Performance vs. S&P500')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()