from functools import partial

from jax_sgmc.util import host_callback


import jax
import jax.numpy as jnp

from jax_sgmc import util
from jax_sgmc import data

# Todo: Test vmap on custom host_callback
