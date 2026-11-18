
import os
import pandas as pd


from config import (
    RESULTS_DIR
)

def load_metrics():


    path = os.path.join(

        RESULTS_DIR,

        "metrics.csv"

    )


    df = pd.read_csv(

        path

    )


    return df





def load_volatility_results():


    path = os.path.join(

        RESULTS_DIR,

        "volatility_analysis.csv"

    )


    df = pd.read_csv(

        path

    )


    return df


def normalize_metrics(df):

    percentage_columns = [

        "Total Return",

        "Annual Return",

        "Maximum Drawdown",

        "Volatility",

        "Win Rate"

    ]


    for column in percentage_columns:


        if column in df.columns:


            df[column] = (

                df[column] * 100

            ).round(2)



    return df



def rank_algorithms(df):

    ranking = df.sort_values(

        by="Sharpe Ratio",

        ascending=False

    )


    ranking["Rank"] = range(

        1,

        len(ranking)+1

    )


    return ranking


def create_summary(metrics):


    summary = metrics[

        [

        "Algorithm",

        "Total Return",

        "Sharpe Ratio",

        "Sortino Ratio",

        "Maximum Drawdown",

        "Calmar Ratio",

        "Volatility",

        "Win Rate",

        "Profit Factor"

        ]

    ]



    return summary


def create_volatility_summary(df):


    summary = df[

        [

        "Algorithm",

        "Volatility",

        "Return",

        "Sharpe",

        "Maximum Drawdown"

        ]

    ]



    return summary


def save(df, filename):


    path = os.path.join(

        RESULTS_DIR,

        filename

    )


    df.to_csv(

        path,

        index=False

    )


    print(

        f"Saved: {path}"

    )



def main():

    print("="*60)

    print("Loading results")

    print("="*60)

    metrics = load_metrics()
    volatility = load_volatility_results()

    metrics = normalize_metrics(
        metrics
    )

    summary = create_summary(
        metrics
    )

    ranking = rank_algorithms(
        metrics
    )

    volatility_summary = create_volatility_summary(
        volatility
    )

    print()
    print("="*60)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*60)
    print(summary)
    print()
    print("="*60)
    print("ALGORITHM RANKING")
    print("="*60)

    print(
        ranking[
        [
        "Rank",
        "Algorithm",
        "Sharpe Ratio"
        ]

        ]

    )



    print()
    print("="*60)
    print("VOLATILITY ANALYSIS")
    print("="*60)

    print(
        volatility_summary
    )



    save(
        summary,
        "final_comparison.csv"
    )


    save(
        ranking,
        "algorithm_ranking.csv"
    )

    save(
        volatility_summary,
        "volatility_summary.csv"
    )




if __name__ == "__main__":

    main()