import jax.numpy as jnp 
import jax.random as jr

from deeponetx.data.data import DatasetDeepONet


def test_data():
    key = jr.PRNGKey(0)
    k_branch, k_trunk, k_output = jr.split(key, num=3)

    # Suppsoe a 2D funtion mapping with 1D initial 
    input_branch = jr.normal(k_branch, shape=(150, 100))
    input_trunk = jr.normal(k_trunk, shape=(100, 1))
    output = jr.normal(k_output, shape=(150, 100))
    batch_size = 10
    data = DatasetDeepONet(input_branch, input_trunk, output, batch_size, key=key)

    dataiter = iter(data)
    ds = next(dataiter)
    assert data[0:10][0].shape == (10, 100) # branch
    assert data[0:10][1].shape == (100, 1) # trunk 
    assert data[0:10][2].shape == (10, 100) # output
    assert ds.input_branch.shape == (batch_size, 100) # branch
    assert ds.input_trunk.shape == (100,1) # trunk
    assert ds.output.shape == (batch_size,100) # output