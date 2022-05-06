.. _likelihood_to_potential:
Compute Potential from Likelihood
==================================

Stochastic Gradient MCMC evaluates the potential and the model
for a multiple of observations or all observations. The likelihood might be
written for only a single sample or a batch of data. Therefore, this module acts
as an interface between the different likelihoods and the integrators.

.. doctest::

  >>> from functools import partial
  >>> import jax.numpy as jnp
  >>> import jax.scipy as jscp
  >>> from jax import random, vmap
  >>> from jax_sgmc import data, potential
  >>> from jax_sgmc.data.numpy_loader import NumpyDataLoader

  >>> mean = random.normal(random.PRNGKey(0), shape=(100, 5))
  >>> data_loader = NumpyDataLoader(mean=mean)
  >>>
  >>> test_sample = {'mean': jnp.zeros(5), 'std': jnp.ones(1)}


Setup Data Loaders
-------------------

For demonstration purposes, we setup a data loader to compute the potential for
a random batch of data as well as for the full dataset.

Stochastic Potential
_____________________

The stochastic potential is an estimation of the true potential. It is
calculated over a small dataset and rescaled to the full dataset.

  >>> batch_init, batch_get = data.random_reference_data(data_loader,
  ...                                                    cached_batches_count=50,
  ...                                                    mb_size=5)
  >>> random_data_state = batch_init()


Full Potential
_______________

In combination with the :mod:`jax_sgmc.data` it is possible to calculate the
true potential over the full dataset.
If we specify a batch size of 3, then the likelihood will be sequentially
calculated over batches with the size 3.


  >>> init_fun, fmap_fun = data.full_reference_data(data_loader,
  ...                                               cached_batches_count=50,
  ...                                               mb_size=3)
  >>> data_state = init_fun()


Unbatched Likelihood
----------------------

In the simplest case, the likelihood and model function only accept a single
observation and parameter set.
Therefore, this module maps the evaluation over the mini-batch or even all
observations by making use of Jax's tools ``map``, ``vmap`` and ``pmap``.

The likelihood can be written for a single observation. The
:mod:`jax_sgmc.potential` module then evaluates the likelihood for a batch of
reference data sequentially via ``map`` or parallel via ``vmap`` or ``pmap``.

  >>> def likelihood(sample, observation):
  ...   likelihoods = jscp.stats.norm.logpdf(observation['mean'],
  ...                                        loc=sample['mean'],
  ...                                        scale=sample['std'])
  ...   return jnp.sum(likelihoods)
  >>> prior = lambda unused_sample: 0.0


Stochastic Potential
______________________

The stochastic potential is computed automatically from the likelihood of a
single observation.

  >>>
  >>> stochastic_potential_fn = potential.minibatch_potential(prior,
  ...                                                         likelihood,
  ...                                                         strategy='map')
  >>> new_random_data_state, random_batch = batch_get(random_data_state, information=True)
  >>> potential_eval, unused_state = stochastic_potential_fn(test_sample, random_batch)
  >>>
  >>> print(round(potential_eval))
  838

Full Potential
_______________

Here, the likelihood written for a single observation can be re-used.

  >>> potential_fn = potential.full_potential(prior, likelihood, strategy='vmap')
  >>>
  >>> potential_eval, (data_state, unused_state) = potential_fn(
  ...   test_sample, data_state, fmap_fun)
  >>>
  >>> print(round(potential_eval))
  707



Batched Likelihood
------------------

Some models already accept a batch of reference data. In this case, the
potential function can be constructed by setting ``is_batched = True``. In this
case, it is expected that the returned likelihoods are a vector with shape
``(N,)``, where N is the batch-size.


  >>> @partial(vmap, in_axes=(None, 0))
  ... def batched_likelihood(sample, observation):
  ...   likelihoods = jscp.stats.norm.logpdf(observation['mean'],
  ...                                        loc=sample['mean'],
  ...                                        scale=sample['std'])
  ...   # Only valid samples contribute to the likelihood
  ...   return jnp.sum(likelihoods)
  >>>


Stochastic Potential
_____________________

To compute the correct potential now, the function needs to know that the
likelihood is batched by setting ``is_batched=True``. The strategy setting
has no meaning anymore and can be kept on the default value.

  >>> stochastic_potential_fn = potential.minibatch_potential(prior,
  ...                                                         batched_likelihood,
  ...                                                         is_batched=True,
  ...                                                         strategy='map')
  >>>
  >>> new_random_data_state, random_batch = batch_get(random_data_state, information=True)
  >>> potential_eval, unused_state = stochastic_potential_fn(test_sample, random_batch)
  >>>
  >>> print(round(potential_eval))
  838
  >>>
  >>> _, (likelihoods, _) = stochastic_potential_fn(test_sample,
  ...                                               random_batch,
  ...                                               likelihoods=True)
  >>>
  >>> print(round(jnp.var(likelihoods)))
  7

Full Potential
__________________

The batched likelihood can also be used to calculate the full potential.

  >>> prior = lambda unused_sample: 0.0
  >>>
  >>> potential_fn = potential.full_potential(prior, batched_likelihood, is_batched=True)
  >>>
  >>> potential_eval, (data_state, unused_state) = potential_fn(
  ...   test_sample, data_state, fmap_fun)
  >>>
  >>> print(round(potential_eval))
  707

Likelihoods with States
------------------------

By setting the argument ``has_state = True``, the likelihood accepts an
additional state as first positional argument. This state should not influence
the results of the computation.

  >>> def stateful_likelihood(state, sample, observation):
  ...   n, mean = state
  ...   n += 1
  ...   new_mean = (n-1)/n * mean + 1/n * observation['mean']
  ...
  ...   likelihoods = jscp.stats.norm.logpdf((observation['mean'] - new_mean),
  ...                                        loc=(sample['mean'] - new_mean),
  ...                                        scale=sample['std'])
  ...   return jnp.sum(likelihoods), (n, new_mean)

.. note::
  If the likelihood is not batched (``is_batched=False``), only the state
  corresponding to the computation with the first sample of the batch is
  returned.

Stochastic Potential
____________________

  >>> potential_fn = potential.minibatch_potential(prior,
  ...                                              stateful_likelihood,
  ...                                              has_state=True)
  >>>
  >>> potential_eval, new_state = potential_fn(test_sample,
  ...                                          random_batch,
  ...                                          state=(jnp.array(2), jnp.ones(5)))
  >>>
  >>> print(round(potential_eval))
  838
  >>> print(f"n: {new_state[0] : d}")
  n:  3

Full Potential
_______________


  >>> full_potential_fn = potential.full_potential(prior,
  ...                                         stateful_likelihood,
  ...                                         has_state=True)
  >>>
  >>> potential_eval, (cache_state, new_state) = full_potential_fn(
  ...   test_sample, data_state, fmap_fun, state=(jnp.array(2), jnp.ones(5)))
  >>>
  >>> print(f"n: {new_state[0] : d}")
  n:  36
