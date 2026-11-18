import os
import sys
import pandas as pd

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from src.environment import CustomStockTradingEnv

from config import (

    DATA_DIR,
    A2C_PARAMS,
    TOTAL_TIMESTEPS,
    DEFAULT_SEED,
    get_a2c_model
)

if len(sys.argv) > 1:
    SEED = int(sys.argv[1])

else:
    SEED = DEFAULT_SEED

def load_training_data():

    path = os.path.join(
        DATA_DIR,
        "train.csv"
    )

    df = pd.read_csv(path)
    return df

def create_environment(df):

    env = CustomStockTradingEnv(
        df=df
    )

    env = Monitor(env)

    return env

def train():

    print("="*60)

    print("Loading training data")

    print("="*60)

    train_df = load_training_data()

    print(train_df.head())
  
    print()
    print("="*60)
    print("Creating A2C environment")
    print("="*60)

    env = create_environment(
        train_df
    )

    print()
    print("="*60)
    print("Initializing A2C")
    print("="*60)

    model = A2C(

        policy="MlpPolicy",
        env=env,
        seed=SEED,
        learning_rate=A2C_PARAMS["learning_rate"],
        n_steps=A2C_PARAMS["n_steps"],
        gamma=A2C_PARAMS["gamma"],
        gae_lambda=A2C_PARAMS["gae_lambda"],
        ent_coef=A2C_PARAMS["ent_coef"],
        verbose=1,
        tensorboard_log="logs/A2C/"
    )

    print()
    print("="*60)
    print("Training A2C")
    print("="*60)

    model.learn(

        total_timesteps=
        TOTAL_TIMESTEPS,
        progress_bar=True

    )

    print()
    print("="*60)
    print("Saving A2C model")
    print("="*60)

    model_path = get_a2c_model(SEED)

    model.save(
        model_path
    )

    print()

    model_path = get_a2c_model(SEED)
    
    model.save(
     model_path
    )


    print()
    print("="*60)
    print("A2C training completed")
    print("="*60)

if __name__ == "__main__":
    train()