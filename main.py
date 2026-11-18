import os


os.system(
"python src/download_data.py"
)

os.system(
"python src/preprocess.py"
)

os.system(
"python src/train_ppo.py"
)

os.system(
"python src/train_dqn.py"
)

os.system(
"python src/train_a2c.py"
)

os.system(
"python src/test_agent.py"
)

os.system(
"python src/metrics.py"
)

os.system(
"python src/volatility.py"
)

os.system(
"python src/compare.py"
)

os.system(
"python src/plots.py"
)