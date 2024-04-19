import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from gymnasium import spaces
from src.simulate import RFQPriceSampler


class RFQEnvironment(gym.Env):
    def __init__(
        self,
        λs,
        Q,
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
        reward_setting="value",
    ):
        super(RFQEnvironment, self).__init__()

        # δ_b, δ_a
        self.action_space = spaces.Box(
            low=-0.2,
            high=0.2,
            shape=(2,),
        )

        # λ_b, λ_a
        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(2,),
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

        if reward_setting not in ["value", "utility", "stable"]:
            raise Exception(
                f"Reward setting must be one of {['value', 'utility', 'stable']}."
            )

        self.reward_setting = reward_setting

        self.λ_b, self.λ_a, self.prices, self.q, self.t = (
            None,
            None,
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

        reward = None

        if self.reward_setting == "value":
            reward = self.v[self.t + 1]
        elif self.reward_setting == "stable":
            reward = self.v[self.t + 1] - np.std(self.v[: self.t + 2])
        elif self.reward_setting == "utility":
            reward = self.u[self.t]
        else:
            raise Exception("Reward function not set correctly.")

        self.t += 1

        return (
            self.obs(),
            reward,
            self.t == (self.rfq_price_sampler.num_time_interval - 1),
            False,
            {},
        )

    def render(self):
        fig = plt.figure()
        gs = fig.add_gridspec(2, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)

        axs[0].plot(self.v[: self.t], label="Reward", color="blue")
        axs[0].legend()
        axs[0].set_title("Reward & Cumulative Reward (1 Episode)")

        cumulative = np.cumsum(self.v[: self.t])
        axs[1].plot(
            cumulative,
            color="darkgreen",
            label=f"Cumulative Reward ({round(cumulative[-1], 3)})",
        )
        axs[1].hlines(
            [cumulative[-1]],
            xmin=0,
            xmax=self.t - 1,
            color="darkgreen",
            linestyle="--",
            alpha=0.5,
        )
        axs[1].legend()

        for ax in axs.flat:
            ax.set(xlabel="Iteration", ylabel="$")

        for ax in axs.flat:
            ax.label_outer()

        plt.show()
