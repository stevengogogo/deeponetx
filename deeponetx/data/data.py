"""
Data type. Inspired from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets/blob/main/Diffusion-reaction/DeepONet_DR.ipynb
"""
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple
import abc 
from torch.utils import data

class DataDeepONet(NamedTuple):
    input_branch:jnp.ndarray
    input_trunk:jnp.ndarray
    output:jnp.array

class DatasetDeepONet(data.Dataset):
    """Data for DeepONet

    Args:
        - `input_branch`: jnp.ndarray # conditioned function (sampled) [nsample, ngrid]
        - `input_trunk`: jnp.ndarray # location y [nsample, locations]
        - `output`: jnp.array # output of operator at location y [nsample, locations]
    """
    def __init__(self, input_branch:jnp.ndarray, input_trunk:jnp.ndarray, output:jnp.array, batch_size:int, *,key:jr.PRNGKey):
        self.input_branch = input_branch # input sample
        self.input_trunk = input_trunk # location
        self.output = output # labeled data evulated at y (solution measurements, BC/IC cond
        self.batch_size = batch_size
        self.key = key

    def __getitem__(self, index):
        """Get subset of data
        """
        return DataDeepONet(self.input_branch[index], self.input_trunk, self.output[index])
        
    def __len__(self):
        """Length of data
        """
        return self.input_branch.shape[0]
     
    def sample(self):
        """Sample data
        """
        self.key, key = jr.split(self.key)
        idx = jr.choice(key, self.input_branch.shape[0], (self.batch_size,), replace=False)
        return self[idx]