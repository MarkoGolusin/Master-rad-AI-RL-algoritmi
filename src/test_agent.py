import os
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C

from src.environment import CustomStockTradingEnv
from src.dqn_enviroment import DQNTradingEnv


from config import (
    DATA_DIR,
    RESULTS_DIR,
    PPO_MODEL,
    DQN_MODEL,
    A2C_MODEL
)


def load_test_data():

    path = os.path.join(
        DATA_DIR,
        "test.csv"
    )

    df = pd.read_csv(
        path
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )


    return df


def create_environment(df, algorithm):

    if algorithm == "DQN":

        env = DQNTradingEnv(
            df=df
        )

    else:

        env = CustomStockTradingEnv(
            df=df
        )

    return env

def load_model(algorithm):

    if algorithm == "PPO":
        model = PPO.load(
            PPO_MODEL
        )

    elif algorithm == "DQN":
        model = DQN.load(
            DQN_MODEL
        )

    elif algorithm == "A2C":
        model = A2C.load(
            A2C_MODEL
        )

    else:
        raise ValueError(
            "Unknown algorithm"
        )

    return model

def test_agent(algorithm):

    print("="*60)
    print(
        f"Testing {algorithm}"
    )
    print("="*60)




    df = load_test_data()
    env = create_environment(
        df,
        algorithm
    )

    # Load trained model
    model = load_model(
        algorithm
    )

    # Reset environment
    reset_result = env.reset()


    if isinstance(reset_result, tuple):
        observation, info = reset_result
    else:
        observation = reset_result

    done = False

    total_reward = 0

  
    # Trading loop

    while not done:

        action, _states = model.predict(
            observation,
            deterministic=True
        )

        result = env.step(
            action
        )

        if len(result) == 5:

            observation, reward, terminated, truncated, info = result

        else:
            observation, reward, terminated, info = result
            truncated = False

        total_reward += reward
        done = terminated or truncated

    print()

    print(
        f"{algorithm} testing finished"
    )

    print(
        "Total reward:",
        total_reward
    )

    save_results(

        env,
        algorithm
    )

def save_results(env, algorithm):

    print(
        "Saving results..."
    )

    # Portfolio value
    if hasattr(
        env,
        "asset_memory"
    ):

        portfolio = pd.DataFrame(

            {
                "portfolio_value":
                env.asset_memory

            }
        )

    elif hasattr(
        env,
        "portfolio_history"
    ):
        portfolio = pd.DataFrame(
            {

                "portfolio_value":
                env.portfolio_history
            }
        )

    else:
        portfolio = pd.DataFrame()

    portfolio_path = os.path.join(

        RESULTS_DIR,
        f"{algorithm}_portfolio.csv"

    )

    portfolio.to_csv(

        portfolio_path,
        index=False

    )

    if hasattr(
        env,
        "actions_memory"
    ):
        actions = pd.DataFrame(
            {
                "action":
                env.actions_memory
            }
        )

    elif hasattr(
        env,
        "action_history"
    ):
        actions = pd.DataFrame(
            {
                "action":
                env.action_history
            }
        )

    else:
        actions = pd.DataFrame()

    actions_path = os.path.join(

        RESULTS_DIR,
        f"{algorithm}_actions.csv"
    )

    actions.to_csv(
        actions_path,
        index=False
    )


    print()
    print("Saved:")
    print(portfolio_path)
    print(actions_path)

def main():

    algorithms = [
        "PPO",
        "DQN",
        "A2C"
    ]

    for algorithm in algorithms:
        test_agent(
            algorithm
        )


if __name__ == "__main__":

    main()