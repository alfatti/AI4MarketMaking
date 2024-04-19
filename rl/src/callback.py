import numpy as np


from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from .eval import evaluate_pricing_agent
from .env import RFQEnvironment
from termcolor import cprint


def env_callback():
    return RFQEnvironment(
        np.array([10.83, 73.03]) / 10,
        np.array(
            [
                [-14.01, 4.37, 4.37, 5.27],
                [19.32, -60.91, 12.54, 29.05],
                [19.32, 12.54, -60.91, 29.05],
                [23.67, 15.00, 15.00, -53.67],
            ]
        ),
        reward_setting="utility",
    )


def env_callback_value():
    return RFQEnvironment(
        np.array([10.83, 73.03]) / 10,
        np.array(
            [
                [-14.01, 4.37, 4.37, 5.27],
                [19.32, -60.91, 12.54, 29.05],
                [19.32, 12.54, -60.91, 29.05],
                [23.67, 15.00, 15.00, -53.67],
            ]
        ),
        reward_setting="value",
    )


class TrainingCallback(BaseCallback):
    def __init__(self, performance_checkpoint_freq, env):
        self.n_calls = 0
        self.performance_checkpoint_freq = performance_checkpoint_freq
        self.env = env

    def _on_step(self):
        if self.num_timesteps % self.performance_checkpoint_freq == 0:
            evaluate_pricing_agent(self.env, self.model, 500)
            cprint(f"\nTimestep: {self.num_timesteps}")
        return True
