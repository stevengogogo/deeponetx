"""
Kernels for Gaussian Process. Constructor for Covariance Functions
"""
import jax 
import jax.numpy as jnp
import equinox as eqx 
import abc 
from jax.experimental import checkify

class AbstractKernel(eqx.Module):
    """Kernel Function"""
    @abc.abstractmethod
    def __call__(self, x:float, y:float):
        raise NotImplementedError

    def __add__(self, k):
        """Sum of two kernels"""
        return SumKernel(self, k)
    
    def __mul__(self, k):
        """Product of two kernels"""
        return ProductKernel(self, k)
    
class SquaredExponential(AbstractKernel):
    length_scale: float
    signal_stddev: float

    @jax.jit
    def __call__(self, x:float, y:float):
        return self.signal_stddev**2 * jnp.exp(- (x-y)**2 / (2*self.length_scale**2))

class RationalQuadratic(AbstractKernel):
    mixture: float # alpha > 0
    length_scale: float
    signal_stddev: float

    def __init__(self, mixture: float, length_scale: float, signal_stddev: float):
        if mixture <= 0:
            raise ValueError(f"mixture must be positive. Got {mixture}")

        self.mixture = mixture
        self.length_scale = length_scale
        self.signal_stddev = signal_stddev
    
    @jax.jit
    def __call__(self, x:float, y:float):
        return self.signal_stddev**2 * (1 + (x-y)**2 / (2*self.mixture*self.length_scale**2))**(-self.mixture)

class SumKernel(AbstractKernel):
    """Kernels summ by two kernels"""
    kernel1: AbstractKernel
    kernel2: AbstractKernel

    @jax.jit
    def __call__(self, x:float, y:float):
        return self.kernel1(x,y) + self.kernel2(x,y)

class ProductKernel(AbstractKernel):
    """Kernels product by two kernels"""
    kernel1: AbstractKernel
    kernel2: AbstractKernel

    @jax.jit
    def __call__(self, x:float, y:float):
        return self.kernel1(x,y) * self.kernel2(x,y)

# Helper function 
def cov_matrix(kernel: AbstractKernel, xs: jnp.ndarray, jitter:float=0.):
    """Covariance matrix
    Arg: 
        kernel (AbstractKernel): Kernel function
        xs (jnp.ndarray): Sensor points
        jitter: small value add to the diagnal of the covariance matrix for numerical stability
    Return: 
        A jnp.ndarray of shape (len(xs), len(xs))
    """
    m_cov = jax.vmap(jax.vmap(kernel, 
                    in_axes=(None, 0)), 
            in_axes=(0,None))(xs, xs)
    m_cov = m_cov + jitter * jnp.eye(len(xs))
    return m_cov
        


