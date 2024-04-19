import numpy as np
import matplotlib.pyplot as plt
import time

from IPython.display import clear_output


def evaluate_pricing_agent(env, model, n=500):
    δ_b = []
    δ_a = []
    cumulative_rewards = []

    for _ in range(n):
        state, _ = env.reset()

        done = False

        while not done:
            action, _ = model.predict(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            δ_b.append(action[0])
            δ_a.append(action[1])
        cumulative_rewards.append(np.cumsum(env.v))

    clear_output(True)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(δ_b, density=True, label="δ_b", bins=10, alpha=0.5, color="green")
    plt.hist(δ_a, density=True, label="δ_a", bins=10, alpha=0.5, color="red")
    plt.xlim((-1.1, 1.1))
    plt.legend()

    cumulative_rewards = np.array(cumulative_rewards)

    x = np.arange(cumulative_rewards.shape[1])
    mean = np.mean(cumulative_rewards, axis=0)
    std_dev = np.std(cumulative_rewards, axis=0)

    plt.subplot(1, 2, 2)
    plt.fill_between(x, mean - std_dev, mean + std_dev, alpha=0.5)
    plt.plot(x, mean, label=f"Average Cumulative Return: ${mean[-1]:,.2f}")
    plt.legend()
    plt.show()


def visualize_simulation(env, model):
    state, _ = env.reset()

    done = False

    δ_b = []
    δ_a = []

    while not done:
        action, _ = model.predict(state)

        δ_b.append(action[0])
        δ_a.append(action[1])

        next_state, reward, done, _, _ = env.step(action)
        state = next_state

        # Update y-data for each subplot (e.g., simulate new data)
        clear_output(True)
        _, axs = plt.subplots(2, 2, figsize=(16, 8))

        axs[0][0].plot(env.v[: env.t], label="Reward")
        axs[0][0].legend()
        axs[0][0].set_xlim(0, env.rfq_price_sampler.num_time_interval)

        axs[0][1].plot(env.prices[: env.t], color="b", label="Price")
        axs[0][1].plot(
            env.prices[: env.t] - np.array(δ_b), color="g", label="Bid Quote"
        )
        axs[0][1].plot(
            env.prices[: env.t] + np.array(δ_a), color="r", label="Ask Quote"
        )
        axs[0][1].legend()
        axs[0][1].set_xlim(0, env.rfq_price_sampler.num_time_interval)

        axs[1][0].plot(env.λ_a[: env.t], label="λ_a")
        axs[1][0].plot(env.λ_b[: env.t], label="λ_b")
        axs[1][0].legend()
        axs[1][0].set_xlim(0, env.rfq_price_sampler.num_time_interval)
        axs[1][0].set_ylim(0, 8)

        axs[1][1].plot((env.λ_a - env.λ_b)[: env.t], label="λ_a - λ_b")
        axs[1][1].hlines(
            [0],
            xmin=0,
            xmax=env.rfq_price_sampler.num_time_interval,
            linestyles="--",
            alpha=0.2,
        )
        axs[1][1].legend()
        axs[1][1].set_xlim(0, env.rfq_price_sampler.num_time_interval)
        axs[1][1].set_ylim(-8, 8)

        plt.show()

        time.sleep(1)
