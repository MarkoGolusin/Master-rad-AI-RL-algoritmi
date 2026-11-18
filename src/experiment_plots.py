import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv(
    "results/experiment_results.csv"
)


df.boxplot(

    column="Return",

    by="Agent"

)


plt.title(
    "Return distribution across seeds"
)


plt.ylabel(
    "Return"
)


plt.savefig(
    "results/plots/return_distribution.png"
)