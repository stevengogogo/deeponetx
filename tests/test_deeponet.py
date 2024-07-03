import pytest 
import equinox as eqx
import jax 
import jax.numpy as jnp
import jax.random as jr
import optax
# test module
from deeponetx import nn
from deeponetx import train

def create_data(key):
    k_branch, k_trunk = jr.split(key, num=2)
    input_branch = jax.random.normal(k_branch, shape=(150, 100))
    input_trunk = jax.random.normal(k_trunk, shape=(100, 1))
    output = jax.random.normal(key, shape=(150, 100))
    return input_branch, input_trunk, output

def test_build_deeponet():
    key = jr.PRNGKey(0)
    k_ney, k_data = jr.split(key, num=2)
    net = nn.create_UnstackDeepONet1d_MLP(100, 4, 4, 40, jax.nn.relu, key=k_ney)

    input_branch, input_trunk, output = create_data(k_data)

    out = jax.vmap(jax.vmap(net, in_axes=(None, 0)), in_axes=(0, None))(input_branch, input_trunk)
    
    assert out.shape == output.shape

def test_training():
    # Create keys
    key = jr.PRNGKey(0)
    k_net, k_data = jr.split(key, num=2)

    # Create net
    net = nn.create_UnstackDeepONet1d_MLP(100, 4, 4, 4, jax.nn.relu, key=k_net)
    
    # Create data
    input_branch, input_trunk, output = create_data(k_data)
    data = train.DataDeepONet(input_branch, input_trunk, output)

    # Train
    train.train(net, data, optax.adam(1e-3), 10)






