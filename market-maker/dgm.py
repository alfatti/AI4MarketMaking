import torch
import torch.nn as nn


def loss(
    θ: nn.Module,
    q: float,
    t: float,
    T: float,
    Q: torch.tensor,
    κ: float,
    γ: float,
    σ: float,
    z: float,
    α_a: torch.tensor,
    α_b: torch.tensor,
    λ_a: float,
    λ_b: float,
) -> float:

    dt = torch.autograd.grad(θ(t, q), t)

    Q_sum = 0
    for j_a in range(4):
        for j_b in range(4):
            Q_sum += Q[j_a][j_b] * θ(t, q)

    return (
        dt
        + κ * (λ_a - λ_b) * q
        - 1 / 2 * γ * σ**2 * q**2
        + Q_sum
        + z * (λ_b * α_b[0] + λ_a * α_a[0])
        + (
            λ_b * α_b[1] * (θ(t, q) - θ(t, q + z))
            + λ_a * α_a[1] * (θ(t, q) - θ(t, q - z))
        )
        + (1 / (2 * z))
        * (
            λ_b * α_b[2] * (θ(t, q) - θ(t, q + z)) ** 2
            + λ_a * α_a[2] * (θ(t, q) - θ(t, q - z)) ** 2
        )
        + (torch.exp(θ(T, q)) - 1)
    )


class NeuralBox(nn.Module):
    def __init__(self, indim=100, outdim=50):
        super().__init__()
        self.activation = nn.Tanh()
        self.z = nn.Linear(outdim, outdim, bias=False)
        self.g = nn.Linear(outdim, outdim, bias=False)
        self.r = nn.Linear(outdim, outdim, bias=False)
        self.h = nn.Linear(outdim, outdim, bias=False)
        self.z1 = nn.Linear(outdim, outdim)
        self.g1 = nn.Linear(outdim, outdim)
        self.r1 = nn.Linear(outdim, outdim)
        self.h1 = nn.Linear(outdim, outdim)
        self.s1 = nn.Linear(indim, outdim)
        self.s2 = nn.Linear(indim, outdim)
        self.s3 = nn.Linear(indim, outdim)
        self.s4 = nn.Linear(indim, outdim)

    def forward(self, x, s):
        z1 = self.z(s) + self.s1(x)
        z1 = self.activation(z1)
        g1 = self.g(s) + self.s2(x)
        g1 = self.activation(g1)
        r1 = self.r(s) + self.s3(x)
        r1 = self.activation(r1)
        h1 = self.h(torch.mul(s, r1)) + self.s4(x)
        h1 = self.activation(h1)
        s2 = torch.mul((1.0 - g1), h1) + torch.mul(z1, s)
        return s2


class DGM(nn.Module):
    def __init__(
        self,
        dim=1,
        layersize=10,
    ):
        """
        We are minimizing the following function (page 28 of the paper):

        0 = ∂_t θ(t, q) + κ(λ_a - λ_b) q - 1 / 2 * γ σ^2 q^2
            + ∑ Q + θ + z(λ α_b0 + λ α_a0)
            ...

        Also, θ(T, q) should be 0 for all q.

        Ahmad wants a standalone loss function that he can use to create the
        NN.

        - m_a and m_b = 4 (the dimensions of Q)
        - ∂θ - compute using PyTorch
        - ...
        """
        super().__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, layersize)
        self.module1 = NeuralBox(indim=self.dim, outdim=layersize)
        self.module2 = NeuralBox(indim=self.dim, outdim=layersize)
        self.module3 = NeuralBox(indim=self.dim, outdim=layersize)

        self.last_layer = nn.Linear(layersize, 1)

        self.activation = nn.Tanh()

    def forward(self, y):
        # model inputs are t and Q
        s1 = self.layer1(y)
        s1 = self.activation(s1)
        s2 = self.module1(y, s1)
        s3 = self.module2(y, s2)
        s4 = self.module3(y, s3)
        out = self.last_layer(s4)
        return out
