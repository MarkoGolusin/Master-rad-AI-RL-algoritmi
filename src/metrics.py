
import os
import numpy as np
import pandas as pd


from config import (
    RESULTS_DIR,
    METRICS_FILE,
    RISK_FREE_RATE
)

# LOAD DATA
def load_portfolio(algorithm):
    path = os.path.join(
        RESULTS_DIR,
        f"{algorithm}_portfolio.csv"
    )
    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Portfolio file not found: {path}"
        )
    df = pd.read_csv(path)
    return df


# DAILY RETURNS

def calculate_returns(df):
    returns = (

        df["portfolio_value"]

        .pct_change()

        .dropna()
    )
    return returns

# TOTAL RETURN

def total_return(df):
    start_value = (

        df["portfolio_value"].iloc[0]
    )
    end_value = (

        df["portfolio_value"].iloc[-1]
    )
    return (
        end_value - start_value
    ) / start_value


# ANNUAL RETURN

def annual_return(df):

    total = total_return(df)
    days = len(df)
    years = days / 252

    if years == 0:
        return 0

    return (

        (1 + total)**(1 / years)
    ) - 1


# SHARPE

def sharpe_ratio(returns):

    if returns.std() == 0:
        return 0
    excess_return = (
        returns.mean()-RISK_FREE_RATE / 252
    )

    return (

        excess_return/returns.std()
    ) * np.sqrt(252)


def sortino_ratio(returns):

    negative_returns = returns[
        returns < 0
    ]

    if len(negative_returns) == 0:
        return 0

    downside_std = (
        negative_returns.std()
    )

    if downside_std == 0:
        return 0

    return (

        returns.mean() / downside_std
    ) * np.sqrt(252)


def maximum_drawdown(df):

    values = (
        df["portfolio_value"]
    )

    running_max = (
        values.cummax()
    )

    drawdown = (

        values - running_max

    ) / running_max

    return drawdown.min()

def calmar_ratio(df):
    annual = annual_return(df)
    mdd = abs(
        maximum_drawdown(df)
    )

    if mdd == 0:
        return 0

    return annual / mdd

def volatility(returns):

    return (
        returns.std()*np.sqrt(252)
    )

def win_rate(returns):

    if len(returns) == 0:

        return 0

    wins = (
        returns > 0
    ).sum()

    return wins / len(returns)

def profit_factor(returns):

    gains = (
        returns[returns > 0]
        .sum()
    )


    losses = abs(
        returns[returns < 0]
        .sum()
    )

    if losses == 0:
        return 0

    return gains / losses

def evaluate_algorithm(algorithm):

    df = load_portfolio(
        algorithm
    )

    returns = calculate_returns(
        df

    )

    metrics = {

        "Algorithm":algorithm,
        "Total Return":total_return(df),
        "Annual Return": annual_return(df),
        "Sharpe Ratio":sharpe_ratio(returns),
        "Sortino Ratio":sortino_ratio(returns),
        "Maximum Drawdown": maximum_drawdown(df),
        "Calmar Ratio":calmar_ratio(df),
        "Volatility": volatility(returns),
        "Win Rate":win_rate(returns),
        "Profit Factor":profit_factor(returns)

    }

    return metrics

def main():

    print("=" * 60)
    print("RL AGENT PERFORMANCE EVALUATION")
    print("=" * 60)

    algorithms = [
        "PPO",
        "DQN",
        "A2C"
    ]

    results = []

    for algorithm in algorithms:
        print()
        print(
            f"Calculating metrics for {algorithm}"
        )

        result = evaluate_algorithm(
            algorithm
        )

        results.append(
            result
        )

    metrics_df = pd.DataFrame(
        results
    )

    metrics_df.to_csv(

        METRICS_FILE,
        index=False

    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        metrics_df.to_string(index=False)

    )

    print()
    print(
        "Metrics saved to:"
    )
    print(
        METRICS_FILE
    )
    print("=" * 60)

if __name__ == "__main__":

    main()