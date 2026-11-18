

import os



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create folders if they do not exist

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)



# Stock ticker (S&P500 company)

TICKER = ["SPY"]



TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2022-01-01"

TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2025-01-01"



INITIAL_AMOUNT = 100000



BUY_COST = [0.001]
SELL_COST = [0.001]



HMAX = 100


TECHNICAL_INDICATORS = [

    "macd",

    "rsi_30",

    "cci_30",

    "dx_30",

    "boll_ub",

    "boll_lb",

    "close_30_sma",

    "close_60_sma",
    

]

STATE_FEATURES = [

    "macd",

    "rsi_30",

    "cci_30",

    "dx_30",

    "boll_ub",

    "boll_lb",

    "close_30_sma",

    "close_60_sma",

    "ATR",

    "volatility"

]


REWARD_SCALING = 1e-4

TOTAL_TIMESTEPS = 100000

PPO_PARAMS = {

    "learning_rate": 0.00025,

    "n_steps": 2048,

    "batch_size": 64,

    "gamma":0.99,

    "gae_lambda":0.95,

    "ent_coef": 0.01

}


def get_ppo_model(seed):

    return os.path.join(
        MODEL_DIR,
        f"ppo_model_seed_{seed}.zip"
    )


DQN_PARAMS = {

    "learning_rate": 0.0005,

    "buffer_size": 50000,

    "learning_starts": 1000,

    "batch_size": 64

}


def get_dqn_model(seed):

    return os.path.join(
        MODEL_DIR,
        f"dqn_model_seed_{seed}.zip"
    )

A2C_PARAMS = {

    "learning_rate": 0.0007,

    "n_steps": 512,

    "gamma": 0.99,

    "gae_lambda": 1.0,

    "ent_coef": 0.01

}


def get_a2c_model(seed):

    return os.path.join(
        MODEL_DIR,
        f"a2c_model_seed_{seed}.zip"
    )

A2C_MODEL = os.path.join(
    MODEL_DIR,
    "a2c_model"
)

SEEDS = [

    42,

    123,

    456,

    789,

    999

]

DEFAULT_SEED = 42

RISK_FREE_RATE = 0.02

VOLATILITY_WINDOW = 20

PLOTS_DIR = os.path.join(
    RESULTS_DIR,
    "plots"
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)

FIGURE_SIZE = (14, 7)

DPI = 150


TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train.csv")

TEST_DATA_FILE = os.path.join(DATA_DIR, "test.csv")

PROCESSED_FILE = os.path.join(DATA_DIR, "processed.csv")

PPO_MODEL = os.path.join(MODEL_DIR, "ppo_model.zip")

DQN_MODEL = os.path.join(MODEL_DIR, "dqn_model.zip")

A2C_MODEL = os.path.join(MODEL_DIR, "a2c_model.zip")

RESULT_TABLE = os.path.join(RESULTS_DIR, "comparison.csv")

METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.csv")

