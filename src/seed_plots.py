import os
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR,
    PLOTS_DIR,
    FIGURE_SIZE,
    DPI
)


def load_results(algorithm):
    file = os.path.join(
        RESULTS_DIR,
        f"{algorithm.lower()}_seed_metrics.csv"
    )

    if not os.path.exists(file):
        return None

    return pd.read_csv(file)


def create_boxplots(df, algorithm):
    metrics = [
        "Total Return",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Volatility"
    ]

    for metric in metrics:
        plt.figure(
            figsize=FIGURE_SIZE
        )

        plt.boxplot(
            df[metric]
        )
        plt.title(
            f"{algorithm} Multi-Seed Distribution - {metric}"
        )

        plt.ylabel(
            metric
        )

        plt.grid(
            True
        )

        output = os.path.join(
            PLOTS_DIR,
            f"{algorithm.lower()}_{metric.replace(' ','_').lower()}_boxplot.png"
        )

        plt.savefig(
            output,
            dpi=DPI,
            bbox_inches="tight"
        )

        plt.close()
        print(
            "Saved:",
            output
        )


def create_return_distribution(df, algorithm):
    plt.figure(
        figsize=FIGURE_SIZE
    )

    plt.hist(
        df["Total Return"],
        bins=min(10, len(df))
    )

    plt.title(
        f"{algorithm} Return Distribution Across Seeds"
    )
    plt.xlabel(
        "Total Return"
    )
    plt.ylabel(
        "Frequency"
    )

    plt.grid(
        True
    )

    output = os.path.join(
        PLOTS_DIR,
        f"{algorithm.lower()}_return_distribution.png"
    )
    plt.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output
    )


def create_seed_comparison(df, algorithm):
    plt.figure(
        figsize=FIGURE_SIZE
    )

    plt.bar(
        df["Seed"].astype(str),
        df["Total Return"]
    )
    plt.title(
        f"{algorithm} Total Return per Seed"
    )

    plt.xlabel(
        "Random Seed"
    )

    plt.ylabel(
        "Total Return"
    )

    plt.grid(
        True,
        axis="y"
    )

    output = os.path.join(
        PLOTS_DIR,
        f"{algorithm.lower()}_seed_comparison.png"
    )

    plt.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output
    )


def create_portfolio_plot(algorithm):
    plt.figure(
        figsize=FIGURE_SIZE
    )

    found = False

    for file in os.listdir(RESULTS_DIR):
        if (
            file.startswith(f"{algorithm}_seed_")
            and
            file.endswith("_portfolio.csv")
        ):
            found = True

            path = os.path.join(
                RESULTS_DIR,
                file
            )

            df = pd.read_csv(path)

            seed = file.split("_")[2]

            plt.plot(
                df.index,
                df["portfolio_value"],
                label=f"Seed {seed}"
            )

    if not found:
        print(
            f"No {algorithm} portfolio files found."
        )

        plt.close()

        return

    plt.title(
        f"{algorithm} Portfolio Value During Testing"
    )

    plt.xlabel(
        "Trading Step"
    )

    plt.ylabel(
        "Portfolio Value"
    )

    plt.legend()

    plt.grid(True)

    output = os.path.join(
        PLOTS_DIR,
        f"{algorithm.lower()}_portfolio_value.png"
    )

    plt.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output
    )


def process_algorithm(algorithm):
    df = load_results(algorithm)

    if df is None:
        print(
            f"{algorithm} metrics not found."
        )

        return

    create_boxplots(
        df,
        algorithm
    )

    create_return_distribution(
        df,
        algorithm
    )

    create_seed_comparison(
        df,
        algorithm
    )

    create_portfolio_plot(
        algorithm
    )


def create_algorithm_comparison():
    algorithms = [
        "PPO",
        "A2C",
        "DQN"
    ]

    data = {}

    for algorithm in algorithms:
        file = os.path.join(
            RESULTS_DIR,
            f"{algorithm.lower()}_seed_metrics.csv"
        )

        if os.path.exists(file):
            data[algorithm] = pd.read_csv(file)

    if len(data) < 2:
        print(
            "Not enough algorithms for comparison."
        )

        return

    metrics = [
        "Total Return",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Volatility"
    ]

    for metric in metrics:
        plt.figure(
            figsize=FIGURE_SIZE
        )

        names = []
        means = []
        stds = []

        for algorithm, df in data.items():
            names.append(
                algorithm
            )

            means.append(
                df[metric].mean()
            )

            stds.append(
                df[metric].std()
            )

        plt.bar(
            names,
            means,
            yerr=stds,
            capsize=8
        )

        plt.ylabel(
            metric
        )

        plt.title(
            f"PPO vs A2C vs DQN - {metric}"
        )

        plt.grid(
            True,
            axis="y"
        )

        output = os.path.join(
            PLOTS_DIR,
            f"comparison_{metric.replace(' ','_').lower()}.png"
        )

        plt.savefig(
            output,
            dpi=DPI,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Saved:",
            output
        )



def create_portfolio_comparison():
    plt.figure(
        figsize=FIGURE_SIZE
    )

    algorithms = [
        "PPO",
        "A2C",
        "DQN"
    ]

    for algorithm in algorithms:
        values = []

        for file in sorted(os.listdir(RESULTS_DIR)):
            if (
                file.startswith(f"{algorithm}_seed_")
                and
                file.endswith("_portfolio.csv")
            ):
                df = pd.read_csv(
                    os.path.join(
                        RESULTS_DIR,
                        file
                    )
                )

                values.append(
                    df["portfolio_value"]
                )

        if len(values) == 0:
            continue

        mean_curve = pd.concat(
            values,
            axis=1
        ).mean(axis=1)

        plt.plot(
            mean_curve,
            linewidth=3,
            label=algorithm
        )

    plt.title(
        "Average Portfolio Value Across Random Seeds"
    )

    plt.xlabel(
        "Trading Step"
    )

    plt.ylabel(
        "Portfolio Value"
    )

    plt.grid(True)

    plt.legend()

    output = os.path.join(
        PLOTS_DIR,
        "comparison_portfolio_value.png"
    )

    plt.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output
    )


def main():
    print("="*60)

    print(
        "MULTI-SEED PPO + A2C + DQN PLOTS"
    )

    print("="*60)

    process_algorithm("PPO")

    process_algorithm("A2C")

    process_algorithm("DQN")

    create_algorithm_comparison()

    create_portfolio_comparison()

    print()

    print("="*60)

    print(
        "PLOTS COMPLETED"
    )

    print("="*60)


if __name__ == "__main__":
    main()