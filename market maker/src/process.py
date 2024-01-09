import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor

from .utils import two_combinations
from .simulate import simulate_markov_batch


class StochasticProcess:
    """
    Base class for defining PDE related function.

    Args:
    eqn_config (dict): dictionary containing PDE configuration parameters

    Attributes:
    dim (int): dimensionality of the problem
    total_time (float): total time horizon
    num_time_interval (int): number of time steps
    delta_t (float): time step size
    sqrt_delta_t (float): square root of time step size
    y_init (None): initial value of the function
    """

    def __init__(self, eqn_config: dict):
        self.dim = eqn_config["dim"]
        self.total_time = eqn_config["total_time"]
        self.num_time_interval = eqn_config["num_time_interval"]
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def __str__(self):
        attributes_str = (
            "{\n"
            + ", \n\n".join(f"{key}: \n{value}" for key, value in vars(self).items())
            + "\n}"
        )

        return f"{self.__class__.__name__}:\n{attributes_str}"

    def sample(self, num_sample: int) -> Tensor:
        """
        Sample forward SDE.

        Args:
        num_sample (int): number of samples to generate

        Returns:
        Tensor: tensor of size [num_sample, dim+1] containing samples
        """
        raise NotImplementedError

    def r_u(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """
        Interest rate in the PDE.

        Args:
        t (float): current time
        x (Tensor): tensor of size [batch_size, dim] containing space coordinates
        y (Tensor): tensor of size [batch_size, 1] containing function values
        z (Tensor): tensor of size [batch_size, dim] containing gradients

        Returns:
        Tensor: tensor of size [batch_size, 1] containing generator values
        """
        raise NotImplementedError

    def h_z(self, t, x, y, z: Tensor) -> Tensor:
        """
        Function to compute H(z) in the PDE.

        Args:
        h (float): value of H function
        z (Tensor): tensor of size [batch_size, dim] containing gradients

        Returns:
        Tensor: tensor of size [batch_size, dim] containing H(z)
        """
        raise NotImplementedError

    def terminal(self, x: Tensor) -> Tensor:
        """
        Terminal condition of the PDE.

        Args:
        t (float): current time
        x (Tensor): tensor of size [batch_size, dim] containing space coordinates

        Returns:
        Tensor: tensor of size [batch_size, 1] containing terminal values
        """
        raise NotImplementedError


class RFQ(StochasticProcess):
    def __init__(self, basic_config, specific_config):
        super().__init__(basic_config)

        self.n_liqiudity_state = specific_config["nls"]

        # initial mid-price for bond 1 in sector 1 from Table 3, page 23 in the Bergault paper
        self.x_init = np.ones(self.dim) * specific_config["init"]

        self.lamda_initial_state = specific_config["init_state"]  # integer

        # sigma and kappa from Table 2, page 23 in the Bergault paper
        self.sigma = specific_config["sigma"]
        self.k = specific_config["k"]

        # lambda and Q values from Table 1, page 20 in the Bergault paper
        self.lamdas = specific_config["lamdas"]
        self.Q = specific_config["Q"]

    def sample(self, num_sample) -> tuple:
        dw_sample = (
            np.random.normal(size=[num_sample, self.num_time_interval])
            * self.sqrt_delta_t
        )

        x_sample = np.zeros([num_sample, self.num_time_interval + 1])
        x_sample[:, 0] = np.ones(num_sample) * self.x_init

        select_Q = np.ones(
            [num_sample, self.n_liqiudity_state**2, self.n_liqiudity_state**2]
        ) * np.expand_dims(np.exp(self.Q * self.delta_t), axis=0)
        select_lamda = np.ones(
            [num_sample, self.n_liqiudity_state**2, 2]
        ) * np.expand_dims(two_combinations(self.lamdas), axis=0)

        lamda_process = simulate_markov_batch(
            self.Q,
            self.lamda_initial_state,
            np.array([i * self.delta_t for i in range(self.num_time_interval)]),
            num_sample,
        )

        ask_lamda = np.array([[self.lamdas[x // 2] for x in y] for y in lamda_process])
        bid_lamda = np.array([[self.lamdas[x % 2] for x in y] for y in lamda_process])

        for i in range(self.num_time_interval):
            x_sample[:, i + 1] = (
                x_sample[:, i]
                + (ask_lamda[:, i] - bid_lamda[:, i]) * self.k * self.delta_t
                + self.sigma * dw_sample[:, i]
            )
        return ask_lamda, bid_lamda, x_sample
