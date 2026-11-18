import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor

from src.environment import CustomStockTradingEnv

from config import (
    DATA_DIR,
    RESULTS_DIR,
    SEEDS,
    get_ppo_model,
    get_a2c_model,
    get_dqn_model
)


def load_test_data():

    path = os.path.join(
        DATA_DIR,
        "test.csv"
    )

    df = pd.read_csv(
        path
    )

    return df

def create_environment(df):
    base_env = CustomStockTradingEnv(
        df=df
    )
    env = Monitor(
        base_env
    )
    return env

def test_agent(
        algorithm,
        seed
):

    print()
    print("="*60)
    print(
        f"Testing {algorithm} seed {seed}"
    )
    print("="*60)

    # SELECT MODEL
    if algorithm == "PPO":
        model_path = get_ppo_model(seed)
        model_class = PPO

    elif algorithm == "A2C":
        model_path = get_a2c_model(seed)
        model_class = A2C

    elif algorithm == "DQN":
        model_path = get_dqn_model(seed)
        model_class = DQN

    else:
        print(
            "Unknown algorithm:",
            algorithm
        )

        return

    if not os.path.exists(model_path):

        print(
            "Model not found:",
            model_path
        )

        return

    df = load_test_data()

    env = create_environment(
        df
    )

    model = model_class.load(
        model_path
    )

    # RUN TEST
    obs, info = env.reset()
    done = False
    total_reward = 0
    portfolio = []
    actions = []
    step = 0

    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True
        )
        # SAVE ACTION
        action_value = np.asarray(action).flatten()[0]

        actions.append(
            float(action_value)
        )

        # SEND ACTION TO ENVIRONMENT
        # FinRL expects numpy array
        # DQN returns scalar
        if algorithm == "DQN":
            action_to_env = np.array(
                [action_value]
            )

        else:
            action_to_env = action

        obs, reward, terminated, truncated, info = env.step(
            action_to_env

        )

        done = terminated or truncated
        total_reward += reward

        portfolio.append(
            env.unwrapped.asset_memory[-1]
        )
        step += 1

        # safety limit
        if step > len(df):

            print(
                "Warning: maximum steps reached"
            )
            break

    # SAVE PORTFOLIO
    portfolio_file = os.path.join(
        RESULTS_DIR,
        f"{algorithm}_seed_{seed}_portfolio.csv"

    )

    portfolio_df = pd.DataFrame(
        {
            "Step": range(len(portfolio)),
            "portfolio_value": portfolio
        }
    )

    portfolio_df.to_csv(
        portfolio_file,
        index=False
    )

    # SAVE ACTIONS
    actions_file = os.path.join(
        RESULTS_DIR,
        f"{algorithm}_seed_{seed}_actions.csv"
    )

    actions_df = pd.DataFrame(
        {
            "action": actions
        }
    )

    actions_df.to_csv(
        actions_file,
        index=False
    )

    print()
    print(f"Total reward: {total_reward}")
    print()
    print("Saved:")
    print(portfolio_file)
    print(
        actions_file
    )


def main():

    print("="*60)
    print(
        "MULTI SEED PPO + A2C + DQN TESTING"
    )
    print("="*60)

    algorithms = [
        "PPO",
        "A2C",
        "DQN"
    ]

    for algorithm in algorithms:
        for seed in SEEDS:
            test_agent(
                algorithm,
                seed
            )

    print()
    print("="*60)
    print(
        "ALL TESTS FINISHED"
    )
    print("="*60)

if __name__ == "__main__":

    main()