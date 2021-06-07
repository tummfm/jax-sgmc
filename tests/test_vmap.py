from functools import partial

from jax.experimental import host_callback


import jax
import jax.numpy as jnp

from jax_sgmc import util
from jax_sgmc import data

test_pytree = {"a": jnp.ones(10), "b": {
  "b1": jnp.zeros(10),
  "b2": jnp.ones(9)
}}

format = data.tree_dtype_struct(test_pytree)

tf, _ = util.pytree_list_transform([test_pytree, test_pytree])

test = False

@partial(jax.vmap, in_axes=0, out_axes=0)
def test_fn(tree):
  value = host_callback.call(lambda x: x, tree, result_shape=format)
  return jax.tree_map(jnp.add, value, test_pytree)


print(test_fn(tf([test_pytree, test_pytree])))