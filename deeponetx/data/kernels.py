"""
Kernels for Gaussian Process. Constructor for Covariance Functions
"""
import jax 
import jax.numpy as jnp
import equinox as eqx 
import abc 
from jax.experimental import checkify

class AbstractKernel(eqx.Module):
    @abc.abstractmethod
    def __call__(self, x:float, y:float):
        raise NotImplementedError

class SquaredExponential(AbstractKernel):
    length_scale: float
    signal_stddev: float

    @eqx.filter_jit
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
    
    @eqx.filter_jit 
    def __call__(self, x:float, y:float):
        return self.signal_stddev**2 * (1 + (x-y)**2 / (2*self.mixture*self.length_scale**2))**(-self.mixture)

class SumKernel(AbstractKernel):
    """Kernels summ by two kernels"""
    kernel1: AbstractKernel
    kernel2: AbstractKernel

    @eqx.filter_jit
    def __call__(self, x:float, y:float):
        return self.kernel1(x,y) + self.kernel2(x,y)

class ProductKernel(AbstractKernel):
    """Kernels product by two kernels"""
    kernel1: AbstractKernel
    kernel2: AbstractKernel

    @eqx.filter_jit
    def __call__(self, x:float, y:float):
        return self.kernel1(x,y) * self.kernel2(x,y)

        


