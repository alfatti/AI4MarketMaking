import numpy as np
import matplotlib.pyplot as plt
import time

from IPython.display import clear_output


def evaluate_pricing_agent(env, model, n=500, clear=True):
    δ_b = []
    δ_a = []
    cumulative_returns = []
    rewards = []

    for _ in range(n):
        reward_run = []
        state, _ = env.reset()

        done = False

        while not done:
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            δ_b.append(action[0])
            δ_a.append(action[1])
            reward_run.append(reward)

        cumulative_returns.append(np.cumsum(env.v))
        rewards.append(reward_run)

    if clear:
        clear_output(True)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.hist(δ_b, label="δ_b", bins=20, alpha=0.5, color="green")
    plt.hist(δ_a, label="δ_a", bins=20, alpha=0.5, color="red")
    plt.xlim((-1.2, 1.2))
    plt.legend()

    cumulative_returns = np.array(cumulative_returns)

    x = np.arange(cumulative_returns.shape[1])
    mean = np.mean(cumulative_returns, axis=0)
    std_dev = np.std(cumulative_returns, axis=0)

    plt.subplot(1, 4, 2)
    plt.fill_between(x, mean - std_dev, mean + std_dev, alpha=0.5)
    plt.plot(x, mean, label=f"Average Cumulative Return: ${mean[-1]:,.2f}")
    plt.legend()

    rewards = np.array(rewards)

    plt.subplot(1, 4, 3)
    plt.plot(rewards[-1, :], label=f"Reward Sample")
    plt.plot(np.cumsum(rewards[-1, :]), label=f"Cumulative Reward Sample")
    plt.legend()

    x = np.arange(rewards.shape[1])
    mean = np.mean(rewards, axis=0)
    std_dev = np.std(rewards, axis=0)

    plt.subplot(1, 4, 4)
    plt.fill_between(x, mean - std_dev, mean + std_dev, alpha=0.5)
    plt.plot(x, mean, label=f"Average Cumulative Rewards: {mean[-1]:,.2f}")
    plt.legend()
    plt.show()


def visualize_simulation(env, model, sleep=1, clear=True):
    state, _ = env.reset()

    done = False

    δ_b = []
    δ_a = []
    rewards = []

    while not done:
        action, _ = model.predict(state, deterministic=True)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state

        δ_b.append(action[0])
        δ_a.append(action[1])
        rewards.append(reward)

        if clear:
            clear_output(True)

        _, axs = plt.subplots(2, 3, figsize=(16, 8))

        axs[0][0].plot(env.q[: env.t], label="Quantity", alpha=0.7)
        axs[0][0].legend()
        axs[0][0].set_xlim(0, env.rfq_price_sampler.num_time_interval)

        axs[0][1].plot(env.prices[: env.t], color="b", label="Price", alpha=0.7)
        axs[0][1].plot(
            env.prices[: env.t] - np.array(δ_b)[: env.t],
            color="g",
            label="Bid Quote",
            alpha=0.7,
        )
        axs[0][1].plot(
            env.prices[: env.t] + np.array(δ_a)[: env.t],
            color="r",
            label="Ask Quote",
            alpha=0.7,
        )
        axs[0][1].legend()
        axs[0][1].set_xlim(0, env.rfq_price_sampler.num_time_interval)

        axs[0][2].plot(rewards, label="Reward", alpha=0.7)
        axs[0][2].plot(np.cumsum(rewards), label="Cumulative Rewards", alpha=0.7)
        axs[0][2].legend()
        axs[0][2].set_xlim(0, env.rfq_price_sampler.num_time_interval)

        axs[1][0].plot((env.λ_a - env.λ_b)[: env.t], label="λ_a - λ_b", alpha=0.7)
        axs[1][0].hlines(
            [0],
            xmin=0,
            xmax=env.rfq_price_sampler.num_time_interval,
            linestyles="--",
            alpha=0.2,
        )
        axs[1][0].legend()
        axs[1][0].set_xlim(0, env.rfq_price_sampler.num_time_interval)
        axs[1][0].set_ylim(-8, 8)

        axs[1][1].plot(δ_b, label="δ_b", color="g", alpha=0.4)
        axs[1][1].plot(δ_a, label="δ_a", color="r", alpha=0.4)
        axs[1][1].legend()
        axs[1][1].set_xlim(0, env.rfq_price_sampler.num_time_interval)
        axs[1][1].set_ylim(-1.2, 1.2)

        axs[1][2].plot(env.v[: env.t], label="Portfolio Value", alpha=0.7)
        axs[1][2].legend()
        axs[1][2].set_xlim(0, env.rfq_price_sampler.num_time_interval)

        plt.show()

        time.sleep(sleep)
