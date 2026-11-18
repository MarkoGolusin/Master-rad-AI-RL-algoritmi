import os
import numpy as np
import pandas as pd


from config import (
    RESULTS_DIR,
    SEEDS
)

def load_seed_results(
        algorithm,
        seed
):

    file = os.path.join(
        RESULTS_DIR,
        f"{algorithm}_seed_{seed}_portfolio.csv"

    )

    if not os.path.exists(file):
        print(
            f"Missing file: {file}"
        )

        return None

    df = pd.read_csv(
        file
    )
    return df

def calculate_metrics(
        algorithm,
        seed
):

    df = load_seed_results(
        algorithm,
        seed
    )

    if df is None:
        return None

    portfolio = df["portfolio_value"]
    initial = portfolio.iloc[0]
    final = portfolio.iloc[-1]

    total_return = (
        final - initial
    ) / initial

    daily_returns = (
        portfolio
        .pct_change()
        .dropna()
    )

    volatility = (
        daily_returns.std()*np.sqrt(252)

    )

    if daily_returns.std() != 0:

        sharpe = (
            daily_returns.mean()/daily_returns.std()
        ) * np.sqrt(252)

    else:
        sharpe = 0


    running_max = (

        portfolio.cummax()
    )


    drawdown = (
        portfolio - running_max
    ) / running_max

    max_drawdown = (
        drawdown.min()

    )

    return {

        "Algorithm": algorithm,
        "Seed": seed,
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown,
        "Volatility": volatility
    }

def calculate_algorithm_metrics(
        algorithm
):

    results = []

    for seed in SEEDS:

        print(
            f"Calculating {algorithm} seed {seed}"

        )

        metric = calculate_metrics(
            algorithm,
            seed
        )

        if metric:
            results.append(
                metric
            )

    df = pd.DataFrame(
        results
    )

    print()
    print("="*60)
    print(f"{algorithm} RESULTS")
    print("="*60)
    print(df)

    if not df.empty:
        print()
        print("="*60)
        print( f"{algorithm} DISTRIBUTION")
        print("="*60)
        print(
            df.describe()
        )

    else:
        print(
            f"No results available for {algorithm}"
        )

    output = os.path.join(
        RESULTS_DIR,
        f"{algorithm.lower()}_seed_metrics.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print()
    print(
        "Saved:"
    )
    print(
        output
    )

def main():

    print("="*60)
    print(
        "MULTI SEED PPO + A2C + DQN METRICS"
    )

    print("="*60)

    algorithms = [
        "PPO",
        "A2C",
        "DQN"
    ]


    for algorithm in algorithms:
        calculate_algorithm_metrics(
            algorithm
        )

    print()
    print("="*60)
    print(
        "METRICS COMPLETED"
    )

    print("="*60)

if __name__ == "__main__":

    main()