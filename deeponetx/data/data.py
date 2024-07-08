"""
Data type. Inspired from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets/blob/main/Diffusion-reaction/DeepONet_DR.ipynb
"""
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple
import abc 

class DataDeepONet(NamedTuple):
    """Data for DeepONet

    Args:
        - `input_branch`: jnp.ndarray # conditioned function (sampled) [nsample, ngrid]
        - `input_trunk`: jnp.ndarray # location y [nsample, locations]
        - `output`: jnp.array # output of operator at location y [nsample, locations]
    """
    input_branch: jnp.ndarray # input sample
    input_trunk: jnp.ndarray # location
    output: jnp.array # labeled data evulated at y (solution measurements, BC/IC cond
    def __getitem__(self, index):
        """Get subset of data
        """
        return DataDeepONet(self.input_branch[index], self.input_trunk, self.output[index])
    
    def sample(self, batch_size:int, key):
        """Sample data
        """
        idx = jr.choice(key, self.input_branch.shape[0], (batch_size,), replace=False)
        return self[idx]