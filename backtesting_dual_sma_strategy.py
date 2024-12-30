import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import random
"""
Assumptions:
- Buy/sell signals are based on the dual moving average crossover strategy.
- Golden Cross: Short SMA crosses above Long SMA.
- Death Cross: Short SMA crosses below Long SMA.
"""

# ****************************** Parameters ******************************
ticker = 'SPY'  # Single stock to analyze
start_date = '2023-01-01'
end_date = '2024-09-30'
short_sma = 5
long_sma = 30
# **************************** DO NOT MODIFY BELOW ****************************

warnings.filterwarnings("ignore", category=RuntimeWarning)

SHORT_SMA_STR = f"SMA{short_sma}"
LONG_SMA_STR = f"SMA{long_sma}"

def calculate_return(prices, start_date, end_date):
    start_price = prices.loc[start_date]
    end_price = prices.loc[end_date]
    return (end_price - start_price) / start_price

def calculate_max_drawdown(prices):
    prices = prices.dropna()
    cumulative_return = prices / prices.iloc[0]
    running_max = cumulative_return.cummax()
    drawdown = cumulative_return / running_max - 1
    return drawdown.min()

def check_golden_death_cross(data, SHORT_SMA_STR, LONG_SMA_STR):
    golden_crosses = []
    death_crosses = []
    for i in range(1, len(data)):
        if not pd.isna(data[SHORT_SMA_STR][i]) and not pd.isna(data[LONG_SMA_STR][i]):
            if data[SHORT_SMA_STR][i] >= data[LONG_SMA_STR][i] and data[SHORT_SMA_STR][i-1] < data[LONG_SMA_STR][i-1]:
                golden_crosses.append(data.index[i])
            elif data[SHORT_SMA_STR][i] <= data[LONG_SMA_STR][i] and data[SHORT_SMA_STR][i-1] > data[LONG_SMA_STR][i-1]:
                death_crosses.append(data.index[i])
    return golden_crosses, death_crosses

# **************************** PART 0: Download Data ****************************
data = yf.download(ticker, start=start_date, end=end_date)

# calculate simple moving averages
data['Close'] = data['Close']/data['Close'].iloc[0] # normalize to ratios
data = pd.DataFrame(data)
data[SHORT_SMA_STR] = data['Close'].rolling(window=short_sma, min_periods=1).mean()
data[LONG_SMA_STR] = data['Close'].rolling(window=long_sma, min_periods=1).mean()

# **************************** PART 1: identify buy/sell signals and price changes ****************************
golden_crosses, death_crosses = check_golden_death_cross(data, SHORT_SMA_STR, LONG_SMA_STR)

# Simulate equity line
equity = np.zeros(len(data))
in_position = False
buy_price = 0.0
initial_price = data['Close'].iloc[0]  # Use the first price as the initial price
cumulative_return = 1.0  # Start with a base value of 1 for cumulative return

# track cumulative return and equity
cumulative_returns = []
cumulative_returns.append(0)
equity[0] = data['Close'].iloc[0]
first_bought_index = None

for i in range(len(data)):
    cumulative_return = 0
    prev_equity = equity[i-1] if i > 0 and equity[i-1] > 0 else 0

    # buy stock
    if data.index[i] in golden_crosses:
        if (first_bought_index == None):
            first_bought_index = data.index[i]
            prev_equity = data['Close'].iloc[i-1]
        in_position = True
        buy_price = data['Close'].iloc[i]

    # sell stock
    elif data.index[i] in death_crosses:
        if first_bought_index != None:
            in_position = False
            sell_price = data['Close'].iloc[i]
            # update cumulative return based on price difference
            cumulative_return = sell_price - buy_price
        else:
            death_crosses.remove(data.index[i])

    if (first_bought_index == None):
        equity[i] = 0
        cumulative_returns.append(0)
        continue

    # calculate the equity value when in position (holding stock)
    if in_position:
        equity[i] = prev_equity + data['Close'].iloc[i] - data['Close'].iloc[i-1] if i > 0 else data['Close'].iloc[0]  # Holding stock
    else:
        equity[i] = prev_equity if i > 0 else data['Close'].iloc[0]  # No stock

    # Update cumulative return after each change
    if in_position:
        cumulative_returns.append(cumulative_returns[len(cumulative_returns)-1])
    else:
        cumulative_returns.append(cumulative_returns[len(cumulative_returns)-1] + cumulative_return)

cumulative_returns.pop(0)

data['Equity'] = equity
data['Cumulative Return'] = cumulative_returns



# **************************** PART 2: random buy/sell signals ****************************
random_indices = sorted(random.sample(range(len(data)), 10))  # Generate 10 random indices
random_signals = [data.index[idx] for idx in random_indices]

random_buys = random_signals[::2]  # odd indices for buys
random_sells = random_signals[1::2]  # even indices for sells

# **************************** calculate Random Returns ****************************
random_equity = np.zeros(len(data))
in_position = False
buy_price = 0.0
initial_price = data['Close'].iloc[0] 
random_cumulative_return = 1.0  

random_cumulative_returns = []
random_cumulative_returns.append(0)
random_equity[0] = data['Close'].iloc[0]
random_first_bought_index = None

for i in range(len(data)):
    random_cumulative_return = 0
    random_prev_random_equity = random_equity[i-1] if i > 0 and random_equity[i-1] > 0 else 0

    # buy stock
    if data.index[i] in random_buys:
        if (random_first_bought_index == None):
            random_first_bought_index = data.index[i]
            random_prev_random_equity = data['Close'].iloc[i-1]
        in_position = True
        buy_price = data['Close'].iloc[i]

    # sell stock
    elif data.index[i] in random_sells:
        in_position = False
        sell_price = data['Close'].iloc[i]
        # update cumulative return based on price difference
        random_cumulative_return = sell_price - buy_price

    if (random_first_bought_index == None):
        random_equity[i] = 0
        random_cumulative_returns.append(0)
        continue

    # calculate the random_equity value when in position (holding stock)
    if in_position:
        random_equity[i] = random_prev_random_equity + data['Close'].iloc[i] - data['Close'].iloc[i-1] if i > 0 else data['Close'].iloc[0]  # Holding stock
    else:
        random_equity[i] = random_prev_random_equity if i > 0 else data['Close'].iloc[0]  # No stock

    # Update cumulative return after each change
    if in_position:
        random_cumulative_returns.append(random_cumulative_returns[len(random_cumulative_returns)-1])
    else:
        random_cumulative_returns.append(random_cumulative_returns[len(random_cumulative_returns)-1] + random_cumulative_return)

random_cumulative_returns.pop(0)

data['Random Equity'] = random_equity
data['Random Cumulative Return'] = random_cumulative_returns



# **************************** PART 3: Plotting Results ****************************

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# subplot 1: golden/death cross buy/sell with SMA
equity_data = data.iloc[data.index >= first_bought_index]

axes[0].plot(data['Close'], label=f'{ticker} Price', color='blue')
axes[0].plot(equity_data['Equity'], label='Equity Line', color='purple', linestyle='--')
axes[0].plot(data[SHORT_SMA_STR], label=SHORT_SMA_STR, color='gold')
axes[0].plot(data[LONG_SMA_STR], label=LONG_SMA_STR, color='coral')
axes[0].scatter(golden_crosses, data.loc[golden_crosses, 'Close'], color='green', label='Buy (Golden Cross)', marker='^', s=100)
axes[0].scatter(death_crosses, data.loc[death_crosses, 'Close'], color='red', label='Sell (Death Cross)', marker='v', s=100)
axes[0].set_title(f'{ticker} Price with Dual SMA Buy/Sell Signals')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price')
axes[0].legend()
axes[0].grid(True)

# subplot 2: random buy/sell
random_equity_data = data.iloc[data.index >= random_first_bought_index]

axes[1].plot(data['Close'], label=f'{ticker} Price', color='blue')
axes[1].plot(random_equity_data['Random Equity'], label='Equity Line', color='purple', linestyle='--')
axes[1].scatter(random_buys, data.loc[random_buys, 'Close'], color='green', label='Random Buy', marker='^', s=100)
axes[1].scatter(random_sells, data.loc[random_sells, 'Close'], color='red', label='Random Sell', marker='v', s=100)
axes[1].set_title(f'{ticker} Price with Random Buy/Sell Signals')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Price')
axes[1].legend()
axes[1].grid(True)

# subplot 3: cumulative returns
cum_return_data = data.iloc[data.index >= first_bought_index]
random_cum_return_data = data.iloc[data.index >= random_first_bought_index]

axes[2].plot(cum_return_data.index, cum_return_data['Cumulative Return'], label='Golden/death crosses', color='slateblue')
axes[2].plot(random_cum_return_data.index, random_cum_return_data['Random Cumulative Return'], label='Random buy/sell', color='orange')
axes[2].set_title('Cumulative Returns Comparison')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Cumulative Return')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()