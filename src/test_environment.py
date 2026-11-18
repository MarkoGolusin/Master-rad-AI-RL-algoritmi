import pandas as pd

from src.environment import CustomStockTradingEnv


df = pd.read_csv(
    "data/train.csv"
)

env = CustomStockTradingEnv(df)
state, info = env.reset()


print("======================")
print("Environment OK")
print("======================")

print(
    "State size:",
    len(state)
)

action = env.action_space.sample()
state, reward, done, truncated, info = env.step(action)

print(
    "Action:",
    action
)

print(
    "Reward:",
    reward
)

print(
    "Done:",
    done
)