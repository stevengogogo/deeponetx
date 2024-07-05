"""
Function spaces for DeepONet-X. Inspired by https://github.com/lululxvi/deepxde/blob/a856b4e2ef5e97e46629ba09f7120b49a40b135f/deepxde/data/function_spaces.py
"""

import abc 
import jax.numpy as np
import diffrax as dfx