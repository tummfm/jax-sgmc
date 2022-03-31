Compute Potential from Likelihood
==================================

Stochastic gradient monte carlo requires to evaluate the potential and the model
for a multiple of observations or all observations. However, the likelihood and
model function only accept a singe observation and parameter set. Therefore,
this module maps the evaluation over the mini-batch or even all observations by
making use of jaxs tools ``map``, ``vmap`` and ``pmap``.

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


Stochastic Potential
---------------------

The stochastic potential is an estimation of the true potential. It is
calculated over a small dataset and rescaled to the full dataset.

  >>> batch_init, batch_get = data.random_reference_data(data_loader,
  ...                                                    cached_batches_count=50,
  ...                                                    mb_size=5)
  >>> random_data_state = batch_init()

Unbatched Likelihood
_____________________

The likelihood can be written for a single observation. The
:mod:`jax_sgmc.potential` module then evaluates the likelihood for a batch of
reference data sequentially via ``map`` or parallel via ``vmap`` or ``pmap``.

  >>> def likelihood(sample, observation):
  ...   likelihoods = jscp.stats.norm.logpdf(observation['mean'],
  ...                                        loc=sample['mean'],
  ...                                        scale=sample['std'])
  ...   return jnp.sum(likelihoods)
  >>> prior = lambda unused_sample: 0.0
  >>>
  >>> stochastic_potential_fn = potential.minibatch_potential(prior,
  ...                                                         likelihood,
  ...                                                         strategy='map')
  >>> new_random_data_state, random_batch = batch_get(random_data_state, information=True)
  >>> potential_eval, unused_state = stochastic_potential_fn(test_sample, random_batch)
  >>>
  >>> print(potential_eval)
  837.9893

Batched Likelihood
___________________

Some models already accept a batch of reference data. In this case, the
potential function can be constructed by setting ``is_batched = True``. In this
case, it is expected that the returned likelihoods are a vectore with shape
``(N,)``, where N is the batch-size.


  >>> @partial(vmap, in_axes=(None, 0))
  ... def batched_likelihood(sample, observation):
  ...   likelihoods = jscp.stats.norm.logpdf(observation['mean'],
  ...                                        loc=sample['mean'],
  ...                                        scale=sample['std'])
  ...   # Only valid samples contribute to the likelihood
  ...   return jnp.sum(likelihoods)
  >>>
  >>> stochastic_potential_fn = potential.minibatch_potential(prior,
  ...                                                         batched_likelihood,
  ...                                                         is_batched=True,
  ...                                                         strategy='map')
  >>>
  >>> new_random_data_state, random_batch = batch_get(random_data_state, information=True)
  >>> potential_eval, unused_state = stochastic_potential_fn(test_sample, random_batch)
  >>>
  >>> print(potential_eval)
  837.9893
  >>>
  >>> _, (likelihoods, _) = stochastic_potential_fn(test_sample,
  ...                                               random_batch,
  ...                                               likelihoods=True)
  >>>
  >>> print(jnp.var(likelihoods))
  7.289153


Full Potential
---------------

In combination with the :mod:`jax_sgmc.data` it is possible to calculate the
true potential over the full dataset.
If we specify a batch size of 3, then the liklihood will be sequentially
calculated over batches with the size 3.


  >>> init_fun, fmap_fun = data.full_reference_data(data_loader,
  ...                                               cached_batches_count=50,
  ...                                               mb_size=3)
  >>> data_state = init_fun()

Unbatched Likelihood
_____________________

Here, the likelihood written for a single observation can be re-used.

  >>> potential_fn = potential.full_potential(prior, likelihood, strategy='vmap')
  >>>
  >>> potential_eval, (data_state, unused_state) = potential_fn(
  ...   test_sample, data_state, fmap_fun)
  >>>
  >>> print(potential_eval)
  707.4376

Bached Likelihood
__________________

The batched likelihood can also be used to calculate the full potential.

  >>> prior = lambda unused_sample: 0.0
  >>>
  >>> potential_fn = potential.full_potential(prior, batched_likelihood, is_batched=True)
  >>>
  >>> potential_eval, (data_state, unused_state) = potential_fn(
  ...   test_sample, data_state, fmap_fun)
  >>>
  >>> print(potential_eval)
  707.4376


Likelihoods with States
------------------------

By setting the argument ``has_state = True``, the likelihood accepts a state
as first positional argument.

  >>> def statefull_likelihood(state, sample, observation):
  ...   n, mean = state
  ...   n += 1
  ...   new_mean = (n-1)/n * mean + 1/n * observation['mean']
  ...
  ...   likelihoods = jscp.stats.norm.logpdf((observation['mean'] - new_mean),
  ...                                        loc=(sample['mean'] - new_mean),
  ...                                        scale=sample['std'])
  ...   return jnp.sum(likelihoods), (n, new_mean)
  >>>
  >>> potential_fn = potential.minibatch_potential(prior,
  ...                                              statefull_likelihood,
  ...                                              has_state=True)
  >>>
  >>> potential_eval, new_state = potential_fn(test_sample,
  ...                                          random_batch,
  ...                                          state=(jnp.array(2), jnp.ones(5)))
  >>>
  >>> print(potential_eval)
  837.9893
  >>> print(new_state)
  (DeviceArray(3, dtype=int32), DeviceArray([0.8914191 , 0.1184448 , 0.7666685 , 0.55906993, 1.1051651 ],            dtype=float32))