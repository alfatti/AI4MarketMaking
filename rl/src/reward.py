import torch


def s_curve(delta, alpha, beta, delta_0):
    return 1 / (1 + torch.exp(alpha + beta / delta_0 * delta))


def utility(
    ask_delta,
    bid_delta,
    q,
    z,
    ask_lamda,
    bid_lamda,
    gamma,
    alpha,
    beta,
    delta_0,
    kappa,
    delta_t,
    sigma,
):
    t = ask_delta.shape[1]

    res = 0
    for i in range(t):
        res += (
            z
            * ask_delta[:, i]
            * ask_lamda[:, i]
            * s_curve(ask_delta[:, i], alpha, beta, delta_0)
            + z
            * bid_delta[:, i]
            * bid_lamda[:, i]
            * s_curve(bid_delta[:, i], alpha, beta, delta_0)
        ) + (
            +kappa * (ask_lamda[:, i] - bid_lamda[:, i]) * q[:, i]
            - gamma / 2 * q[:, i] ** 2 * sigma**2
        ) * delta_t
    return torch.mean(res)
