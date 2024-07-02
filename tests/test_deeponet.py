import pytest 
import equinox as eqx
import jax 
import jax.numpy as jnp
import jax.random as jr
# test module
from deeponetx import deeponet


def test_build_deeponet():
    key = jr.PRNGKey(0)
    k_ney, k_branch, k_trunk = jr.split(key, num=3)
    net = deeponet.create_UnstackDeepONet1d_MLP(100, 4, 4, 40, jax.nn.relu, key=k_ney)

    input_branch = jax.random.normal(k_branch, shape=(150, 100))
    input_trunk = jax.random.normal(k_trunk, shape=(100, 1))
    output = jax.random.normal(key, shape=(150, 100))

    out = jax.vmap(jax.vmap(net, in_axes=(None, 0)), in_axes=(0, None))(input_branch, input_trunk)
    
    assert out.shape == output.shape