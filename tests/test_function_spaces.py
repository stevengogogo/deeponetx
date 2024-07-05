import jax.numpy as jnp
import jax.random as jr
import deeponetx.data.function_spaces as fs 
import deeponetx.data.kernels as kernels

def test_function_spaces():
    key = jr.PRNGKey(0)
    k = kernels.SquaredExponential(0.5, 1.0)
    x = jnp.linspace(0,1,num=1000)
    grf = fs.GaussianRandomField(k, x, jitter=1e-4)

    grf.random(10, key=key)
    