import os
import numpy as np
import pandas as pd


from config import (
    DATA_DIR,
    RESULTS_DIR,
    VOLATILITY_WINDOW
)


def load_test_data():

    path = os.path.join(
        DATA_DIR,
        "test.csv"
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df

def create_volatility_regimes(df):

    print(
        "Creating volatility regimes..."
    )


    # If volatility already exists
    # from preprocess.py
    if "volatility" not in df.columns:

        df["daily_return"] = (
            df["close"]
            .pct_change()
        )

        df["volatility"] = (
            df["daily_return"].rolling(VOLATILITY_WINDOW).std()
        )


    threshold = (

        df["volatility"]
        .median()
    )

    df["volatility_regime"] = np.where(

        df["volatility"] >= threshold,
        "HIGH",
        "LOW"
    )

    return df


def load_agent_results(algorithm):

    path = os.path.join(
        RESULTS_DIR,
        f"{algorithm}_portfolio.csv"
    )

    portfolio = pd.read_csv(
        path
    )
    return portfolio

def merge_results(portfolio, market):

    length = min(
        len(portfolio),
        len(market)
    )

    portfolio = portfolio.iloc[:length]
    market = market.iloc[:length]
    result = pd.DataFrame()

    result["portfolio_value"] = (
        portfolio["portfolio_value"]
        .values
    )

    result["volatility_regime"] = (
        market["volatility_regime"].values
    )
    return result


def calculate_return(df):

    return (
        df["portfolio_value"].iloc[-1]-df["portfolio_value"].iloc[0]

    ) / df["portfolio_value"].iloc[0]

def calculate_sharpe(df):

    returns = (
        df["portfolio_value"]
        .pct_change()
        .dropna()
    )

    if returns.std() == 0:
        return 0

    return (
        returns.mean()/returns.std()
    ) * np.sqrt(252)

def calculate_drawdown(df):

    values = df["portfolio_value"]
    maximum = values.cummax()
    drawdown = (
        values - maximum
    ) / maximum
    return drawdown.min()


def analyze_algorithm(algorithm, market):

    portfolio = load_agent_results(
        algorithm
    )

    data = merge_results(
        portfolio,
        market
    )

    results = []

    for regime in [
        "LOW",
        "HIGH"
    ]:

        subset = data[
            data["volatility_regime"]
            == regime
        ]

        if len(subset) < 2:
            continue

        results.append(
            {

            "Algorithm":algorithm,
            "Volatility":regime,
            "Return":calculate_return(subset),
            "Sharpe":calculate_sharpe(subset),
            "Maximum Drawdown":calculate_drawdown(subset)
            }
        )

    return results

def main():

    market = load_test_data()
    market = create_volatility_regimes(
        market
    )

    algorithms = [
        "PPO",
        "DQN",
        "A2C"
    ]

    results = []

    for algorithm in algorithms:

        print(
            f"Analyzing {algorithm}"
        )

        output = analyze_algorithm(
            algorithm,
            market
        )

        results.extend(
            output
        )

    df = pd.DataFrame(
        results
    )
    path = os.path.join(
        RESULTS_DIR,
        "volatility_analysis.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print()
    print("="*60)
    print(df)
    print("="*60)

    print("Saved:")
    print(path)

if __name__ == "__main__":

    main()