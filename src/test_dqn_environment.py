import os
import pandas as pd

from src.dqn_enviroment import DQNTradingEnv

from config import DATA_DIR

def load_data():

    path = os.path.join(
        DATA_DIR,
        "train.csv"
    )

    df = pd.read_csv(path)
    return df

def main():

    print("======================")
    print("DQN Environment Test")
    print("======================")

    df = load_data()

    env = DQNTradingEnv(
        df
    )

    print()
    print("Environment OK")
    print()
    print(
        "State size:",
        env.observation_space.shape
    )

    reset = env.reset()

    if isinstance(reset, tuple):
        state, info = reset
    else:
        state = reset

    print()
    print(
        "Initial state:"
    )
    print(
        state
    )


    # DQN actions
    actions = [

        0,  # HOLD
        1,  # BUY
        2   # SELL

    ]



    for action in actions:

        print()
        print(
            "Action:",
            action
        )
        state, reward, terminated, truncated, info = env.step(
            action
        )
        print(
            "Reward:",
            reward
        )
        print(
            "Done:",
            terminated or truncated
        )

if __name__ == "__main__":

    main()