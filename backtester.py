
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------
uso = yf.download('USO', start='2015-01-01', end='2024-12-31', progress=False)['Close']
usl = yf.download('USL', start='2015-01-01', end='2024-12-31', progress=False)['Close']

df = pd.DataFrame({'USO': uso.squeeze(), 'USL': usl.squeeze()}).dropna()
print(f"Data Shape: {df.shape}")

# ---------------------------------------------------------------------
# 2. Roll-yield signal (long backwardation, flat contango)
# ---------------------------------------------------------------------
df['Roll_Proxy'] = df['USO'] / df['USL']
df['Roll_Normalized'] = df['Roll_Proxy'] / df['Roll_Proxy'].rolling(60).mean() - 1
df['Smoothed_Roll'] = df['Roll_Normalized'].rolling(5).mean()
df['Signal'] = np.where(df['Smoothed_Roll'] > 0, 1, 0)
df['Position'] = df['Signal'].shift(1).fillna(0)

# ---------------------------------------------------------------------
# 3. Options tail-risk hedge (Black-Scholes put pricing)
# ---------------------------------------------------------------------
df['Market_Return'] = df['USO'].pct_change()
df['Hist_Vol'] = df['Market_Return'].rolling(30).std() * np.sqrt(252)

def bs_put_price(S, K, T, r, sigma):
    sigma = np.where(sigma <= 0, 1e-6, sigma)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

df['Strike_Price'] = df['USO'] * 0.95
df['Put_Premium_Pct'] = bs_put_price(
    S=df['USO'], K=df['Strike_Price'], T=30/365, r=0.05, sigma=df['Hist_Vol']
) / df['USO']
df['Hedge_Drag'] = np.where(df['Position'] == 1, df['Put_Premium_Pct'] / 21, 0)

# ---------------------------------------------------------------------
# 4. Costs and net returns
# ---------------------------------------------------------------------
df['Gross_Return'] = df['Position'] * df['Market_Return']
df['Trades'] = df['Position'].diff().abs().fillna(0)
T_COST_BPS = 0.0005

df['Net_Return'] = df['Gross_Return'] - (df['Trades'] * T_COST_BPS) - df['Hedge_Drag']
df['BnH_Return'] = df['Market_Return']

# ---------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------
def calc_metrics(returns_series):
    clean_returns = returns_series.dropna()
    sharpe = (clean_returns.mean() / clean_returns.std()) * np.sqrt(252)
    cumulative = (1 + clean_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return sharpe, drawdown.min(), cumulative

strat_sharpe, strat_dd, strat_cum = calc_metrics(df['Net_Return'])
bnh_sharpe, bnh_dd, bnh_cum = calc_metrics(df['BnH_Return'])

print("-" * 30)
print(f"Strategy Sharpe (Full Period):  {strat_sharpe:.2f}")
print(f"Strategy Max Drawdown:          {strat_dd:.1%}")
print("-" * 30)
print(f"Buy & Hold Sharpe:              {bnh_sharpe:.2f}")
print(f"Buy & Hold Max Drawdown:        {bnh_dd:.1%}")
print("-" * 30)

# ---------------------------------------------------------------------
# 6. In-sample / out-of-sample validation (no re-fitting after split)
# ---------------------------------------------------------------------
SPLIT_DATE = '2020-12-31'
is_df = df.loc[:SPLIT_DATE]
oos_df = df.loc['2021-01-01':]

is_sharpe, is_dd, _ = calc_metrics(is_df['Net_Return'])
oos_sharpe, oos_dd, _ = calc_metrics(oos_df['Net_Return'])

print("\n--- IN-SAMPLE (2015-2020) ---")
print(f"IS Strategy Sharpe:   {is_sharpe:.2f}")
print(f"IS Strategy Max DD:   {is_dd:.1%}")

print("\n--- OUT-OF-SAMPLE (2021-2024) ---")
print(f"OOS Strategy Sharpe:  {oos_sharpe:.2f}")
print(f"OOS Strategy Max DD:  {oos_dd:.1%}")

# ---------------------------------------------------------------------
# 7. Regime check — was OOS outperformance just the 2022 backwardation spike?
# ---------------------------------------------------------------------
print("\n--- REGIME CHECK ---")
print(df.loc['2021-01-01':'2024-12-31', 'Smoothed_Roll'].describe())
print(f"% time long in 2022: {df.loc['2022-01-01':'2022-12-31', 'Position'].mean():.1%}")
print(f"% time long in 2023: {df.loc['2023-01-01':'2023-12-31', 'Position'].mean():.1%}")

# ---------------------------------------------------------------------
# 8. Equity curve chart
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(strat_cum.index, strat_cum.values, label='Strategy', linewidth=1.5)
ax.plot(bnh_cum.index, bnh_cum.values, label='Buy & Hold (USO)', linewidth=1.5, alpha=0.7)
ax.axvline(pd.Timestamp(SPLIT_DATE), color='grey', linestyle='--', linewidth=1, label='Train/Test Split')
ax.set_title('Roll-Yield Strategy vs Buy & Hold — Cumulative Return (2015-2024)')
ax.set_ylabel('Cumulative Return (1 = starting value)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('equity_curve.png', dpi=150)
print("\nSaved equity_curve.png")