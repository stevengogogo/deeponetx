"""
Define PDE. Inspired from https://github.com/lululxvi/deepxde/blob/master/deepxde/data/pde.py
"""
import jax.numpy as jnp
from .data import Data

class PDE(Data):
    """ODE or time-independent PDE solver"""
    