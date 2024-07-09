import jax.numpy as jnp 
import jax.random as jr

from deeponetx.data.data import DataDeepONet


def test_data():
    key = jr.PRNGKey(0)
    k_branch, k_trunk, k_output = jr.split(key, num=3)

    # Suppsoe a 2D funtion mapping with 1D initial 
    input_branch = jr.normal(k_branch, shape=(150, 1))
    input_trunk = jr.normal(k_trunk, shape=(150, 2))
    output = jr.normal(k_output, shape=(150, 1))

    data = DataDeepONet(input_branch, input_trunk, output)
    ds = data.sample(10, key)
    assert data[0:10].input_branch.shape == (10, 1)
    assert data[0:10].input_trunk.shape == (150, 2)
    assert data[0:10].output.shape == (10, 1)
    assert ds.input_branch.shape == (10, 1)
    assert ds.input_trunk.shape == (150, 2)
    assert ds.output.shape == (10, 1)
