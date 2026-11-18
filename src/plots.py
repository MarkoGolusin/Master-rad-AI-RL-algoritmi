
import os

import pandas as pd
import matplotlib.pyplot as plt


from config import (
    RESULTS_DIR,
    PLOTS_DIR
)



def create_directory():

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )



def get_algorithms():

    algorithms = []


    for file in os.listdir(RESULTS_DIR):

        if file.endswith("_portfolio.csv"):

            name = file.replace(
                "_portfolio.csv",
                ""
            )

            algorithms.append(name)


    return algorithms


def plot_portfolio_growth():

    algorithms = get_algorithms()

    if len(algorithms) == 0:

        print(
            "No portfolio files found"
        )

        return

    plt.figure(
        figsize=(12,6)
    )

    for algorithm in algorithms:
        path = os.path.join(

            RESULTS_DIR,

            f"{algorithm}_portfolio.csv"

        )

        df = pd.read_csv(path)

        plt.plot(

            df["portfolio_value"],

            label=algorithm

        )


    plt.title(
        "Portfolio Value During Testing"
    )


    plt.xlabel(
        "Trading Days"
    )


    plt.ylabel(
        "Portfolio Value ($)"
    )


    plt.legend()
    plt.grid()

    plt.savefig(

        os.path.join(

            PLOTS_DIR,

            "portfolio_growth.png"

        ),

        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def calculate_drawdown(values):
    maximum = values.cummax()
    return (

        values - maximum

    ) / maximum


def plot_drawdown():

    algorithms = get_algorithms()
    plt.figure(
        figsize=(12,6)
    )


    for algorithm in algorithms:

        path = os.path.join(

            RESULTS_DIR,

            f"{algorithm}_portfolio.csv"
        )

        df = pd.read_csv(path)
        dd = calculate_drawdown(

            df["portfolio_value"]
        )

        plt.plot(
            dd,
            label=algorithm

        )

    plt.title(
        "Maximum Drawdown"
    )


    plt.xlabel(
        "Trading Days"
    )


    plt.ylabel(
        "Drawdown"
    )


    plt.legend()

    plt.grid()



    plt.savefig(

        os.path.join(

            PLOTS_DIR,

            "drawdown.png"

        ),

        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_sharpe():


    path = os.path.join(
        RESULTS_DIR,
        "metrics.csv"

    )

    if not os.path.exists(path):

        print(
            "metrics.csv not found"
        )

        return

    df = pd.read_csv(path)

    plt.figure(
        figsize=(8,5)
    )

    plt.bar(

        df["Algorithm"],
        df["Sharpe Ratio"]

    )

    plt.title(
        "Sharpe Ratio Comparison"
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.ylabel(
        "Sharpe Ratio"
    )

    plt.grid(
        axis="y"
    )


    plt.savefig(

        os.path.join(

            PLOTS_DIR,

            "sharpe_comparison.png"

        ),

        dpi=300,
        bbox_inches="tight"

    )

    plt.close()


def plot_volatility():

    path = os.path.join(
        RESULTS_DIR,
        "volatility_analysis.csv"
    )

    if not os.path.exists(path):
        print(
            "No volatility analysis file. Skipping."
        )

        return

    df = pd.read_csv(path)

    pivot = df.pivot(

        index="Algorithm",

        columns="Volatility",

        values="Sharpe"

    )

    pivot.plot(

        kind="bar",

        figsize=(10,6)

    )


    plt.title(
        "Sharpe Ratio During Volatility Regimes"
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.ylabel(
        "Sharpe Ratio"
    )

    plt.grid(
        axis="y"
    )

    plt.legend(
        title="Market Condition"
    )

    plt.savefig(

        os.path.join(

            PLOTS_DIR,

            "volatility_comparison.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


def main():


    print("="*60)

    print(
        "Generating plots..."
    )

    print("="*60)

    create_directory()

    print(
        "Available agents:",
        get_algorithms()
    )

    plot_portfolio_growth()
    plot_drawdown()
    plot_sharpe()
    plot_volatility()

    print()
    print(
        "Plots created successfully"
    )
    print(
        PLOTS_DIR
    )


if __name__ == "__main__":

    main()