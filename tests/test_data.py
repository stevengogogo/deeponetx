import jax.numpy as jnp 
import jax.random as jr

from deeponetx.data.data import DatasetDeepONet


def test_data():
    key = jr.PRNGKey(0)
    k_branch, k_trunk, k_output = jr.split(key, num=3)

    # Suppsoe a 2D funtion mapping with 1D initial 
    input_branch = jr.normal(k_branch, shape=(150, 1))
    input_trunk = jr.normal(k_trunk, shape=(150, 2))
    output = jr.normal(k_output, shape=(150, 1))
    batch_size = 10
    data = DatasetDeepONet(input_branch, input_trunk, output, batch_size, key=key)
    ds = data.sample()
    assert data[0:10][0].shape == (10, 1) # branch
    assert data[0:10][1].shape == (150, 2) # trunk 
    assert data[0:10][2].shape == (10, 1) # output
    assert ds[0].shape == (batch_size, 1) # branch
    assert ds[1].shape == (150, 2) # trunk
    assert ds[2].shape == (batch_size, 1) # branch