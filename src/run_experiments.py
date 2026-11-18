import os
import pandas as pd

from stable_baselines3 import PPO

from src.environment import CustomStockTradingEnv


from config import (
    DATA_DIR,
    RESULTS_DIR,
    TOTAL_TIMESTEPS,
    SEEDS
)


def load_data():
    path = os.path.join(
        DATA_DIR,
        "train.csv"
    )
    return pd.read_csv(path)


def train_seed(seed):

    print("="*60)
    print(
        f"Running seed {seed}"
    )

    print("="*60)

    df = load_data()

    env = CustomStockTradingEnv(
        df=df
    )

    model = PPO(

        "MlpPolicy",
        env,
        seed=seed,
        verbose=0
    )

    model.learn(
        total_timesteps=
        TOTAL_TIMESTEPS
    )

    model.save(

        f"models/PPO_seed_{seed}"
    )

    return model


def main():

    results=[]

    for seed in SEEDS:
        model=train_seed(seed)
        results.append(
            {
            "seed":seed
            }
        )

    df=pd.DataFrame(results)

    df.to_csv(

        os.path.join(

            RESULTS_DIR,
            "experiment_seeds.csv"
        ),

        index=False
    )



if __name__=="__main__":

    main()