import torch
import numpy as np

from .utils import two_combinations


class RFQPriceSampler:
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
    ):
        # λ and Q values from Table 1, page 20 in the Bergault paper
        self.λs = λs
        self.Q = Q

        self.dim = dim
        self.total_time = total_time
        self.num_time_interval = num_time_interval

        self.dt = self.total_time / self.num_time_interval
        self.sqrt_dt = np.sqrt(self.dt)
        self.y_init = None

        self.n_liqiudity_state = n_liquidity_state

        # initial mid-price for bond 1 in sector 1 from Table 3, page 23 in the Bergault paper
        self.x_init = np.ones(self.dim) * init_mid_price

        # the starting state for λ
        self.init_λ_state = init_λ_state

        # σ and κ from Table 2, page 23 in the Bergault paper
        self.σ = σ
        self.κ = κ

    def sample(self) -> tuple:
        # 4 states
        λ_process = self.simulate_markov_batch(
            np.array([i * self.dt for i in range(self.num_time_interval)]),
        )

        # 0: 0-0, 1: 0-1, 2: 1-0, 3: 1-1
        λ_b = np.array([self.λs[x % 2] for x in λ_process])
        λ_a = np.array([self.λs[x // 2] for x in λ_process])

        dw_sample = np.random.normal(size=[self.num_time_interval]) * self.sqrt_dt

        price_sample = np.zeros([self.num_time_interval])
        price_sample[0] = self.x_init

        for i in range(self.num_time_interval - 1):
            price_sample[i + 1] = (
                price_sample[i]
                + (λ_a[i] - λ_b[i]) * self.κ * self.dt
                + self.σ * dw_sample[i]
            )

        return λ_b, λ_a, price_sample, λ_process

    def simulate_markov_batch(self, times):
        num_states = len(self.Q)
        num_times = len(times)

        states_at_times = np.zeros(num_times, dtype=int)

        state = self.init_λ_state
        states_at_times[0] = state

        q = self.Q * self.dt
        I = np.identity(num_states)

        p = np.zeros((1, num_states))
        p[0, state] = 1

        for i in range(1, num_times):
            p = np.matmul(p, q + I)
            state = np.random.choice(num_states, p=np.squeeze(p))
            states_at_times[i] = state

        return states_at_times
