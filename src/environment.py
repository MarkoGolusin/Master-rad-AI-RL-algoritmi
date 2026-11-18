import numpy as np
import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from config import (

    INITIAL_AMOUNT,
    BUY_COST,
    SELL_COST,
    HMAX,
    REWARD_SCALING,
    TECHNICAL_INDICATORS

)



class CustomStockTradingEnv(StockTradingEnv):

    def __init__(self, df):
        self.df = df.copy()

        stock_dimension = len(
            df.tic.unique()
        )

        # STATE SIZE
        state_space = (

             1
             +
             2 * stock_dimension
             +
             len(TECHNICAL_INDICATORS)

)
        super().__init__(
            df=df,
            stock_dim=stock_dimension,
            hmax=HMAX,
            initial_amount=INITIAL_AMOUNT,
            num_stock_shares=[0]*stock_dimension,
            buy_cost_pct=BUY_COST,
            sell_cost_pct=SELL_COST,
            reward_scaling=REWARD_SCALING,
            state_space=state_space,
            action_space=stock_dimension,
            tech_indicator_list=TECHNICAL_INDICATORS,
            print_verbosity=100

        )

        # LOGGING
        self.portfolio_history = []
        self.reward_history = []
        self.action_history = []
        self.volatility_history = []

    def step(self, actions):

        state, reward, terminated, truncated, info = super().step(actions)


        self.reward_history.append(
            reward
        )


        self.action_history.append(
            actions
        )


        if len(self.asset_memory) > 0:

            self.portfolio_history.append(
                self.asset_memory[-1]
            )


        try:

            row = self.df.iloc[self.day]

            self.volatility_history.append(
                row["volatility"]
            )


        except Exception:

            self.volatility_history.append(
                np.nan
            )


        return (
            state,
            reward,
            terminated,
            truncated,
            info
        )

    def reset(self, seed=None, options=None):

        self.portfolio_history = []

        self.reward_history = []

        self.action_history = []

        self.volatility_history = []


        state = super().reset()


        return state

    # LOG DATA
    def get_portfolio_history(self):
        return pd.DataFrame(
            {
                "portfolio_value":

                self.portfolio_history
            }

        )

    def get_rewards(self):
        return pd.DataFrame(
            {
                "reward":
                self.reward_history
            }
        )

    def get_actions(self):
        return pd.DataFrame(
            {
                "action":
                self.action_history
            }
        )

    def get_volatility(self):

        return pd.DataFrame(
            {
                "volatility":
                self.volatility_history
            }
        )