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

def test_function_spaces2d():
    key = jr.PRNGKey(0)
    k = kernels.SquaredExponential(10., 1.0)
    x = jnp.linspace(0,100,num=100)
    y = jnp.linspace(0,100,num=len(x))
    xv, yv = jnp.meshgrid(x,y)
    X = jnp.stack([xv.ravel(), yv.ravel()], axis=1)
    grf = fs.GaussianRandomField(k, X, jitter=1e-3)
    us = grf.sample(10, key=key)

    us = us.reshape(10, len(x), len(y))
    # Visualization
    fig, ax = plt.subplots()
    c = ax.imshow(us[0,:,:])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel("u (sampled function)")
    fig.savefig("tests/img/test_grf2d.png")