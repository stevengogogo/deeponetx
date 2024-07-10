"""
Function spaces for DeepONet-X. Inspired by https://github.com/lululxvi/deepxde/blob/a856b4e2ef5e97e46629ba09f7120b49a40b135f/deepxde/data/function_spaces.py
"""
import jax 
import jax.numpy as jnp
import abc 
import jax.numpy as np
import equinox as eqx
from jax.experimental import enable_x64
from .kernels import AbstractKernel, cov_matrix

class FunctionSpace(eqx.Module):
    @abc.abstractmethod 
    def sample(self, size:int, *, key):
        """Generate feature vectors of random functions

        Args:
            size (int): numer of random functions to generate
            key (_type_): _description_

        Returns:
            A jnp.array of shape (`size`, n_features)
        """
        raise NotImplementedError

class GaussianRandomField(FunctionSpace):
    kernel: AbstractKernel
    xs: np.ndarray # sensor points
    L: np.ndarray # lower choleksy decomp of covariance matrix
    dim: int # dimension of the function space
    N: int # number of sensor points

    def __init__(self, kernel:AbstractKernel, xs:jnp.ndarray, jitter:float=1e-13):
        """_summary_

        Args:
            kernel (AbstractKernel): _description_
            xs (jnp.ndarray): shape ([(coordinate), npoints]). for 1D shape (npoints,); for 2D shape (npoints, 2)
            jitter (float, optional): _description_. Defaults to 1e-13.

        Raises:
            ValueError: _description_
        """
        self.kernel = kernel 
        self.xs = xs 
        self.dim = len(xs.shape)
        self.N = xs.shape[0]
        with enable_x64():
            xs_ = xs.astype(jnp.float64)
            K = cov_matrix(kernel, xs_, jitter=jitter) # covariance matrix
            L = jnp.linalg.cholesky(K, upper=False)

        L = L.astype(jnp.float32)
        if jnp.isnan(L).any():
            raise ValueError("Cholesky decomposition failed")
        self.L = L
    
    def sample(self, n_func:int, *, key):
        """Generate samples from random functions

        Args:
            n_func (int): number of functions to generate
            key (_type_): _description_

        Returns:
            A jnp.array of shape (`n_func`, len(xs))
        """
        u = jax.random.normal(key, shape=(self.N,n_func))
        return (self.L @ u).T