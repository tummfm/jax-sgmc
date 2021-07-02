# from jax import make_jaxpr, jit, vmap
#
# from jax_sgmc.util import host_callback
# from jax_sgmc.util.stop_vmap import stop_vmap
#
# import jax.numpy as jnp
#
#
#
# idx = 0
# def print_var(*var):
#   global idx
#   for v in var:
#     print(f"{idx} ~ Variable: {v}")
#   idx += 1
#
# # Test
#
# # def stop_vmap(fun, *args):
# #   print_var(*args)
# #   if hasattr(args[0], 'batch_dim'):
# #     print("Function gets vmapped")
# #   return fun(*args)
#
# @stop_vmap
# def fun(a, b):
#   host_callback.id_tap(print_var, 0.0)
#   a = a
#   b = b
#   print(a)
#   # assert a.shape == tuple()
#   return {"a": a, "b": [a, b]}
#
# @vmap
# def super_fun(a, b):
#   return fun(a, b)
#
# # print(make_jaxpr(super_fun)(1, 2))
# # print(jit(lambda x: stop_vmap(fun, x, 2))(2))
# # print(super_fun(1, 2))
# #
# # print(make_jaxpr(super_fun)(1, 2))
#
# print("\n\n== Vmap =====\n")
# print(make_jaxpr(super_fun)(jnp.ones(30), jnp.zeros(30)))
# print(jit(super_fun)(jnp.ones(30), jnp.zeros(30)))