"""
Function spaces for DeepONet-X. Inspired by https://github.com/lululxvi/deepxde/blob/a856b4e2ef5e97e46629ba09f7120b49a40b135f/deepxde/data/function_spaces.py
"""
import jax 
import jax.numpy as jnp
import abc 
import jax.numpy as np
import diffrax as dfx
import equinox as eqx
from .kernels import AbstractKernel, cov_matrix

class FunctionSpace(eqx.Module):
    @abc.abstractmethod 
    def random(self, size:int, *, key):
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
    L: np.ndarray # covariance matrix

    def __init__(self, kernel:AbstractKernel, xs:jnp.ndarray, jitter:float=1e-4):
        self.kernel = kernel 
        self.xs = xs 
        K = cov_matrix(kernel, xs, jitter=jitter) # covariance matrix
        self.L = jnp.linalg.cholesky(K, upper=False)
    
    def random(self, n_func:int, *, key):
        """Generate samples from random functions

        Args:
            n_func (int): number of functions to generate
            key (_type_): _description_

        Returns:
            A jnp.array of shape (`n_func`, len(xs))
        """
        u = jax.random.normal(key, shape=(len(self.xs),n_func))
        return jnp.dot(self.L.T, u).T