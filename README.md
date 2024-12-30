# Portfolio optimization
## Project summary
### Part 1: Portfolio optimization
- Implemented **Markowitz mean-variance optimization** to generate **efficient frontier** of multiple tickers
- Plot of randomly-distributed weights with corresponding **Sharpe ratios**
- Determined **equal allocation**, **minimum risk**, and **tangency** portfolios and the corresponding **capital market line**
- Output total return, volatility, max drawdown, and performance in comparison to S&P 500
  
### Part 2: Generating trade strategy
- Backtested performance of SPY stock based on buy/sell signals using the **dual moving average crossover strategy** (short SMA of 5 days, long SMA of 30 days)
- Compared against randomly generated buy/sell signals for the same stock and period
- Displays the equity and culmulative returns of both the dual moving average crossever and random buy/sell strategies

![image](https://github.com/user-attachments/assets/82cfec14-b41b-4329-b2bd-436c6d3602b9)
Optimal portfolio outperformed S&P 500 by 11.47%:
![image](https://github.com/user-attachments/assets/8f5c4f89-a13f-418e-a57c-ede3623b87c5)
![image](https://github.com/user-attachments/assets/5873b7a4-e220-4756-bb55-13e4c34191dd)
![image](https://github.com/user-attachments/assets/16e2d955-1742-4161-a482-653685974608)
![image](https://github.com/user-attachments/assets/c3deb773-1957-43c0-b49c-22a5cc44f143)

Random buy/sell sometimes results in higher return than dual moving average crossover:
![image](https://github.com/user-attachments/assets/7fd993fb-e6fa-425a-b091-e5ba4a169b1f)
