
import numpy as np
import gymnasium as gym


from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


from config import (

    INITIAL_AMOUNT,

    BUY_COST,

    SELL_COST,

    HMAX,

    REWARD_SCALING,

    TECHNICAL_INDICATORS,

    STATE_FEATURES

)



class DQNTradingEnv(StockTradingEnv):

    """
    FinRL environment za DQN.

    Original FinRL:
        Continuous actions


    Modified:

        Discrete actions


    Actions:

        0 = HOLD
        1 = BUY
        2 = SELL

    """



    def __init__(self, df):


        stock_dimension = len(
            df.tic.unique()
        )

        # STATE SPACE
        # Same structure as PPO environment

        state_space = (

            1                    # cash

            +

            2 * stock_dimension # price + shares

            +

            + len(TECHNICAL_INDICATORS)
            * stock_dimension

        )



        super().__init__(


            df=df,


            stock_dim=stock_dimension,


            hmax=HMAX,


            initial_amount=INITIAL_AMOUNT,


            num_stock_shares=[0] * stock_dimension,


            buy_cost_pct=BUY_COST,


            sell_cost_pct=SELL_COST,


            reward_scaling=REWARD_SCALING,


            state_space=state_space,


            # FinRL internal action dimension

            action_space=stock_dimension,


            tech_indicator_list=TECHNICAL_INDICATORS,


            print_verbosity=1000

        )

        # DQN ACTION SPACE
        self.action_space = gym.spaces.Discrete(3)

        # LOGGING
        self.action_history = []

        self.portfolio_history = []

        self.reward_history = []


    # Convert DQN action

    def convert_action(self, action):

        """
        Convert discrete DQN action
        u FinRL continuous action.
        HOLD -> 0
        BUY  -> +1
        SELL -> -1

        """
        if action == 0:
            return np.array([0])
        elif action == 1:
            return np.array([1])
        elif action == 2:
            return np.array([-1])
        else:
            return np.array([0])


    def step(self, action):


        # Save DQN action

        self.action_history.append(
            int(action)
        )

        # Convert action
        converted_action = self.convert_action(

            action

        )

        state, reward, terminated, truncated, info = super().step(

            converted_action

        )

        self.reward_history.append(

            reward

        )

        if len(self.asset_memory) > 0:
            self.portfolio_history.append(

                self.asset_memory[-1]

            )
        return (

            state,
            reward,
            terminated,
            truncated,
            info
        )

    # RESET

    def reset(self, seed=None, options=None):


        self.action_history = []

        self.reward_history = []

        self.portfolio_history = []



        return super().reset(

            seed=seed,

            options=options

        )

    # EXPORT DATA

    def get_actions(self):
        return self.action_history





    def get_portfolio_value(self):
        return self.portfolio_history

    def get_rewards(self):
        return self.reward_history