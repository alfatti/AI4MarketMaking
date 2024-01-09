import gym

from .process import RFQ


class TradingEnv(gym.Env):
    def __init__(self, basic_config, specific_config):
        self.sp = RFQ(basic_config, specific_config)

    def sample(self, batch_size):
        return self.sp.sample(batch_size)
