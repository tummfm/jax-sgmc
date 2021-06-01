#
#
# import jax.numpy as jnp
# from jax.scipy.stats import norm
# from jax import random
#
# from jax_sgmc.integrator import langevin_diffusion
# # from jax_sgmc.data import PreloadReferenceData
# from jax_sgmc.potential import stochastic_potential_gradient
# from jax_sgmc.scheduler import StaticScheduler
#
# class TestLangevinDiffusion:
#   @staticmethod
#   def initialize():
#
#     # A simple normal likelihood with and uniform prior
#
#     def likelihood(sample, parameters, observations):
#       like = norm.logpdf(observations, loc=sample["loc"], scale=sample["scale"])
#       return jnp.sum(like)
#
#     def prior(sample):
#       return jnp.float32(0.0)
#
#     # The integrator requires a potential evaluation funtion which takes care
#     # of handling batches of reference data
#
#     potential_fn = stochastic_potential_gradient(prior, likelihood, 'vmap')
#
#
#     # The reference data is sampled normally
#
#     size = 2
#     observation_count = 10
#
#     key = random.PRNGKey(0)
#     observations = random.normal(key, shape=(observation_count, size)) + 10.0
#
#     reference_data = PreloadReferenceData(observations, batch_size=5)
#
#     # The inital sample is simply set to zero
#
#     init_sample = {"loc": jnp.zeros(2), "scale": jnp.float32(1.0)}
#
#     return init_sample, potential_fn, reference_data
#
#   def test_integrator(self):
#
#     init_sample, potential_fn, reference_data = TestLangevinDiffusion.initialize()
#
#     init_state, integrate = langevin_diffusion(init_sample,
#                                                reference_data,
#                                                potential_fn)
#
#     step_size = 0.01 * jnp.ones(10)
#     temperature = jnp.ones(10)
#
#     scheduler = StaticScheduler(step_size, temperature)
#     schedule = scheduler.get_schedule(10)
#
#     new_state = integrate(init_state, schedule)
#
#     print(init_state)
#     print(new_state)