import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import deeponetx.data.function_spaces as fs 
import deeponetx.data.kernels as kernels

def test_function_spaces():
    key = jr.PRNGKey(0)
    k = kernels.SquaredExponential(0.1, 1.0) 
    x = jnp.linspace(0,1,num=1000)
    grf = fs.GaussianRandomField(k, x, jitter=1e-13)
    us = grf.sample(10, key=key)

    # Visualization
    fig, ax = plt.subplots()
    for i in range(10):
        ax.plot(x, us[i,:])
    ax.set_xlabel("x")
    ax.set_ylabel("u (sampled function)")
    ax.set_title("Sample GRF")
    fig.savefig("tests/img/test_grf.png")