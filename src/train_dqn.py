
import os
import sys
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.dqn_enviroment import DQNTradingEnv

from config import (
    DATA_DIR,
    DQN_PARAMS,
    TOTAL_TIMESTEPS,
    DEFAULT_SEED,
    get_dqn_model
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

    env = DQNTradingEnv(
        df=df
    )
    env = Monitor(env)
    return env

def train():

    print("=" * 60)
    print("Loading training data")
    print("=" * 60)

    train_df = load_training_data()
    print(train_df.head())

    print()
    print("=" * 60)
    print("Creating DQN environment")
    print("=" * 60)

    env = create_environment(
        train_df
    )

    print()
    print("=" * 60)
    print("Initializing DQN")
    print("=" * 60)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        seed=SEED,
        learning_rate=DQN_PARAMS["learning_rate"],
        buffer_size=DQN_PARAMS["buffer_size"],
        learning_starts=DQN_PARAMS["learning_starts"],
        batch_size=DQN_PARAMS["batch_size"],
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log="logs/DQN/"
    )

    print()
    print("=" * 60)
    print("Training DQN")
    print("=" * 60)

    model.learn(
        total_timesteps=
        TOTAL_TIMESTEPS,
        progress_bar=True
    )

    print()
    print("=" * 60)
    print("Saving DQN model")
    print("=" * 60)

    model_path = get_dqn_model(SEED)

    model.save(
        model_path
    )

    print()
    print("Model saved:")
    print(model_path)

    print()
    print("=" * 60)
    print("DQN training completed")
    print("=" * 60)

if __name__ == "__main__":

    train()