import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_DIR, RESULTS_DIR, PLOTS_DIR





INITIAL_AMOUNT = 100000

TEST_FILE = os.path.join(DATA_DIR, "test.csv")

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "buy_hold_portfolio.csv"
)

PLOT_FILE = os.path.join(
    PLOTS_DIR,
    "buy_hold.png"
)


# UČITAVANJE PODATAKA


df = pd.read_csv(TEST_FILE)

# Datum
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

# Provera kolone close
if "close" not in df.columns:
    raise ValueError(
        "Kolona 'close' ne postoji u test.csv"
    )



# BUY & HOLD
initial_price = df["close"].iloc[0]

# Broj kupljenih akcija
shares = INITIAL_AMOUNT / initial_price

# Vrednost portfolija svakog dana
df["portfolio_value"] = df["close"] * shares

# Ukupan prinos
total_return = (
    df["portfolio_value"].iloc[-1] / INITIAL_AMOUNT
) - 1



# DNEVNI PRINOSI
df["daily_return"] = df["portfolio_value"].pct_change()

daily_returns = df["daily_return"].dropna()


# SHARPE

risk_free_rate = 0.02

daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

excess_returns = daily_returns - daily_rf

sharpe = (
    np.sqrt(252)
    * excess_returns.mean()
    / excess_returns.std()
)

# MAX DRAWDOWN
rolling_max = df["portfolio_value"].cummax()

drawdown = (
    df["portfolio_value"] / rolling_max
) - 1

max_drawdown = drawdown.min()


# VOLATILNOST
volatility = daily_returns.std() * np.sqrt(252)



# GODIŠNJI PRINOS
number_of_years = (
    (df["date"].iloc[-1] - df["date"].iloc[0]).days
    / 365.25
)

annual_return = (
    (1 + total_return) ** (1 / number_of_years)
) - 1



# CALMAR
if max_drawdown != 0:
    calmar = annual_return / abs(max_drawdown)
else:
    calmar = np.nan



# ČUVANJE REZULTATA
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

result_df = df.copy()

columns_to_save = [
    "date",
    "close",
    "portfolio_value",
    "daily_return"
]

result_df[columns_to_save].to_csv(
    RESULT_FILE,
    index=False
)



# ISPIS REZULTATA
print("\n" + "=" * 50)
print("BUY & HOLD - SPY")
print("=" * 50)

print(f"Initial portfolio:  ${INITIAL_AMOUNT:,.2f}")
print(f"Initial SPY price:  ${initial_price:.2f}")
print(f"Final SPY price:    ${df['close'].iloc[-1]:.2f}")
print(f"Final portfolio:    ${df['portfolio_value'].iloc[-1]:,.2f}")

print(f"\nTotal Return:       {total_return:.6f}")
print(f"Annual Return:      {annual_return:.6f}")
print(f"Sharpe Ratio:       {sharpe:.6f}")
print(f"Max Drawdown:       {max_drawdown:.6f}")
print(f"Volatility:         {volatility:.6f}")
print(f"Calmar Ratio:       {calmar:.6f}")

print("\nPercentages:")
print(f"Total Return:       {total_return * 100:.2f}%")
print(f"Annual Return:      {annual_return * 100:.2f}%")
print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
print(f"Volatility:         {volatility * 100:.2f}%")

print("\nResults saved to:")
print(RESULT_FILE)



# GRAFIK
trading_steps = np.arange(len(df))

plt.figure(figsize=(12, 6))

plt.plot(
    trading_steps,
    df["portfolio_value"],
    label="Buy & Hold"
)

plt.title("Buy & Hold Portfolio Value - SPY")
plt.xlabel("Trading Step")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=300
)

plt.show()

print("\nPlot saved to:")
print(PLOT_FILE)

plt.savefig(
    PLOT_FILE,
    dpi=300
)

plt.show()

print("\nPlot saved to:")
print(PLOT_FILE)