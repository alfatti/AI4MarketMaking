import gym

from .process import RFQ


class TradingEnv(gym.Env):
    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day;
    def __init__(self, basic_config, specific_config, num_sim=100):
        # simulated data: array of asset price, option price and delta paths (num_path x num_period)
        # generate data now
        self.sp = RFQ(basic_config, specific_config)

    def sample(self, batch_size):
        return self.sp.sample(batch_size)
