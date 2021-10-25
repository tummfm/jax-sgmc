"""Stop vectorization and execute function sequential. """

import logging

def stop_vmap(wrapped):
  logging.warning("vmap of chains is currently not supported")
  return wrapped

# from jax import core, make_jaxpr, api_util
# from jax.tree_util import tree_unflatten, tree_flatten
# from jax import linear_util as lu
# from jax.api_util import flatten_fun_nokwargs
# from jax.linear_util import wrap_init
# from jax.interpreters import partial_eval as pe
# from jax.interpreters import batching
# from jax._src.util import safe_map, partial, extend_name_stack
# from jax._src import abstract_arrays, ad_util
# from jax import lax
#
# try:
#   from jax._src.lib import xla_bridge as xb
# except ModuleNotFoundError:
#   from jax.lib import xla_bridge as xb
#
# from jax.interpreters import xla
# from jax.lib import xla_client
#
#
# xops = xla_client.ops
#
# from jax import jit, vmap
# from jax import numpy as jnp
#
#
# stop_vmap_p = core.Primitive("stop_vmap")
# stop_vmap_p.multiple_results = True
#
# def stop_vmap(wrapped):
#   def wrapper(*args, **kwargs):
#     fun = partial(wrapped, **kwargs)
#     args_flat, in_tree = tree_flatten(args)
#     in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
#     wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
#     jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals)
#     outs = stop_vmap_p.bind(*args_flat, jaxpr=jaxpr, consts=consts)
#     return tree_unflatten(out_tree(), outs)
#   return wrapper
#
# def stop_vmap_decorator(fun):
#   def new_fun(*args):
#     return stop_vmap(fun, *args)
#   return new_fun
#
# def stop_vmap_prim(*args, jaxpr, consts):
#   # Just return the function evaluation
#   return core.eval_jaxpr(jaxpr, consts, *args)
#
# def stop_vmap_abstract_eval(*arg_types, jaxpr, consts):
#   out = [x.aval for x in jaxpr.outvars]
#   return safe_map(core.raise_to_shaped, out)
#
# def stop_vmap_translation_rule(c, axis_env, name_stack, avals, backend,
#                            *args, jaxpr, consts):
#
#   def make_computation(name, jaxpr, op_shape):
#     c = xb.make_computation_builder(name + '_comp')
#     op = xb.parameter(c, 0, op_shape)
#     ops = [xops.GetTupleElement(op, i) for i in range(len(jaxpr.invars))]
#     outs = xla.jaxpr_subcomp(c, jaxpr, backend, axis_env,
#                              safe_map(partial(xb.constant, c), jaxpr.constvars),
#                              extend_name_stack(name_stack, name + '_fun'), *ops)
#     return c.build(xops.Tuple(c, outs))
#
#   op = xops.Tuple(c, args)
#   op_shape = c.get_shape(op)
#   branch_computations = [
#     make_computation(f'branch_{i}', jaxpr, op_shape)
#     for i, jaxpr in enumerate([jaxpr])]
#   return xops.Call(c, branch_computations[0], [op])
#
# def stop_vmap_batching_rule(vector_args, batch_axes, jaxpr, consts):
#   def _helper(args):
#     return stop_vmap_p.bind(*args, jaxpr=jaxpr, consts=consts)
#   res = lax.map(_helper, vector_args)
#   for ax in batch_axes:
#     assert ax == 0, "Can only batch over first axes"
#   return res, [0] * len(res)
#
# stop_vmap_p.def_impl(stop_vmap_prim)
# stop_vmap_p.def_abstract_eval(stop_vmap_abstract_eval)
# xla.initial_style_translations[stop_vmap_p] = stop_vmap_translation_rule
# batching.primitive_batchers[stop_vmap_p] = stop_vmap_batching_rule
#
