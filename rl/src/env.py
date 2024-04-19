import gymnasium as gym
import numpy as np

from gymnasium import spaces
from src.simulate import RFQPriceSampler


class RFQEnvironment(gym.Env):
    def __init__(
        self,
        λs,
        Q,
        δ_min=-0.16,  # gives a 99.8% chance of winning the trade
        δ_max=0.2,  # gives a 99.8% chance of losing the trade
        dim=1,
        total_time=0.25,
        num_time_interval=90,
        n_liquidity_state=2,
        init_mid_price=103.593,
        init_λ_state=1,
        σ=0.1839,
        κ=2.29,
        δ_0=0.09,
        α=-0.7,
        β=3.1,
        γ=0.0005,
        z=1,
    ):
        super(RFQEnvironment, self).__init__()

        self.δ_min = δ_min
        self.δ_max = δ_max

        # δ_b, δ_a
        self.action_space = spaces.Box(
            low=self.δ_min,
            high=self.δ_max,
            shape=(2,),
        )

        # λ_b, λ_a, q, t
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
        )

        self.rfq_price_sampler = RFQPriceSampler(
            λs,
            Q,
            dim,
            total_time,
            num_time_interval,
            n_liquidity_state,
            init_mid_price,
            init_λ_state,
            σ,
            κ,
        )

        self.σ = σ
        self.κ = κ

        self.δ_0 = δ_0
        self.α = α
        self.β = β
        self.γ = γ
        self.z = z

        self.λ_b = None  # λ_b is the number of people that are willing to sell to us (we are buying)
        self.λ_a = None  # λ_a is the number of people that are trying to buy bonds from us (we are selling)

        self.prices, self.q, self.t = (
            None,
            None,
            None,
        )

    def s_curve(self, delta):
        return 1 / (1 + np.exp(self.α + self.β / self.δ_0 * delta))

    def obs(self):
        return np.array(
            [
                self.λ_b[self.t],
                self.λ_a[self.t],
                self.q[self.t],
                self.t / self.rfq_price_sampler.num_time_interval,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None):
        self.t = 0
        (
            self.λ_b,
            self.λ_a,
            self.prices,
            _,
        ) = self.rfq_price_sampler.sample()

        self.q = np.zeros(self.rfq_price_sampler.num_time_interval)
        self.v = np.zeros(self.rfq_price_sampler.num_time_interval)
        self.u = np.zeros(self.rfq_price_sampler.num_time_interval)

        return self.obs(), {}

    def step(self, action):
        n_bid = (
            self.s_curve(action[0]) * self.λ_b[self.t] * self.z
        )  # expected number of bonds we are going to buy
        n_ask = (
            self.s_curve(action[1]) * self.λ_a[self.t] * self.z
        )  # expected number of bonds we are going to sell

        self.q[self.t + 1] = self.q[self.t] + n_bid - n_ask

        bid_price = self.prices[self.t] - action[0]  # Sb_t = S_t - δ_b
        ask_price = self.prices[self.t] + action[1]  # Sa_t = S_t + δ_a

        bid_cost = bid_price * n_bid  # cost of the bonds we are going to buy
        ask_revenue = ask_price * n_ask  # revenue from the bonds we are going to sell

        previous_value = self.q[self.t] * self.prices[self.t]
        next_value = self.q[self.t + 1] * self.prices[self.t + 1]

        self.u[self.t] = (
            self.z * self.λ_b[self.t] * action[0] * self.s_curve(action[0])
            + self.z * self.λ_a[self.t] * action[1] * self.s_curve(action[1])
            + self.κ * (self.λ_a[self.t] - self.λ_b[self.t]) * self.q[self.t]
            - self.γ / 2 * self.σ**2 * self.q[self.t] ** 2
        )

        self.v[self.t + 1] = (
            self.v[self.t] - bid_cost + ask_revenue + (next_value - previous_value)
        )

        reward = self.u[self.t]
        self.t += 1

        return (
            self.obs(),
            reward,
            self.t == (self.rfq_price_sampler.num_time_interval - 1),
            False,
            {},
        )


class RFQEnvironmentNormalized(RFQEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def convert_value(self, input_value):
        low = -1
        high = 1

        # Map input value to the range [0, 1]
        scaled_value = (input_value - low) / (high - low)

        return self.δ_min + scaled_value * (
            self.δ_max - self.δ_min
        )  # Map scaled value to the new range [δ_min, δ_max]

    def step(self, action):
        return super().step(
            [self.convert_value(action[0]), self.convert_value(action[1])]
        )
