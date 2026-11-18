
import os
import sys
import pandas as pd


from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environment import CustomStockTradingEnv


from config import (
    DATA_DIR,
    MODEL_DIR,
    PPO_PARAMS,
    TOTAL_TIMESTEPS,
    DEFAULT_SEED,
    get_ppo_model
)

if len(sys.argv) > 1:
    SEED = int(sys.argv[1])

else:
    SEED = DEFAULT_SEED

def load_training_data():

    file = os.path.join(
        DATA_DIR,
        "train.csv"
    )

    df = pd.read_csv(file)

    return df


def create_environment(df):

    env = CustomStockTradingEnv(df)
    env = Monitor(env)

    return env


def train():

    print("=" * 60)
    print("Loading training data...")
    print("=" * 60)

    train_df = load_training_data()
    print(train_df.head())

    print()
    print("=" * 60)
    print("Creating environment...")
    print("=" * 60)

    env = create_environment(train_df)

    print()
    print("=" * 60)
    print("Initializing PPO...")
    print("=" * 60)

    model = PPO(

        policy="MlpPolicy",
        env=env,
        seed=SEED,
        learning_rate=PPO_PARAMS["learning_rate"],
        batch_size=PPO_PARAMS["batch_size"],
        n_steps=PPO_PARAMS["n_steps"],
        ent_coef=PPO_PARAMS["ent_coef"],
        verbose=1,
        tensorboard_log="logs/PPO/"
    )

    print()
    print("=" * 60)
    print("Training PPO...")
    print("=" * 60)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True
    )

    print()

    print("=" * 60)
    print("Saving model...")
    print("=" * 60)

    model_path = get_ppo_model(SEED)

    model.save(
     model_path
    )

    print()

    print("Model saved to:")
    print(model_path)
    print()

    print("=" * 60)
    print("Training completed.")
    print("=" * 60)


if __name__ == "__main__":

    train()