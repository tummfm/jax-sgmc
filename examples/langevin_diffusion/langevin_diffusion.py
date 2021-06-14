
import numpy as onp
import time

import jax
import jax.numpy as jnp

from jax import random, vmap

import tensorflow

import matplotlib.pyplot as plt

import numpyro

from jax.scipy.stats import norm
from jax.scipy.stats import uniform

from functools import partial

from jax_sgmc import scheduler
from jax_sgmc import potential
from jax_sgmc import integrator
from jax_sgmc import data

#%% md

# Todo: This is not the correct form to handle batches? -> It is possible but it
# is kind of multiple batching

N = 4
M = 100

samples = 1000

sigma = 0.5

key = random.PRNGKey(0)
split, key = random.split(key)

w = random.uniform(split, minval=-1, maxval=1, shape=(N, 1))

def draw_samples(key, w):
    # For each sample draw a new set of predictor variables x and calculate
    # the response

    split1, split2, key = random.split(key, 3)

    noise = jnp.sqrt(sigma) * random.normal(split2, shape=(samples, 1))
    x = random.uniform(split1, minval=-10, maxval=10, shape=(samples, N))
    x = jnp.stack([x[:, 0] + x[:, 1], x[:, 1], 0.1 * x[:, 2] - 0.5 * x[:, 3],
                   x[:, 3]]).transpose()

    y = jnp.matmul(x, w) + noise

    return x, y

# Bring in some correlations

X, Y = draw_samples(key, w)


ds = tensorflow.data.Dataset.from_tensor_slices({"x": X, "y": Y})

data_loader = data.NumpyDataLoader(M, x=X, y=Y)
# data_loader = data.TensorflowDataLoader(ds, mini_batch_size=M, shuffle_cache=1000)

batch_fn = data.random_reference_data(data_loader, cached_batches_count=999)
# First we need to define the (deterministic) model

def model(sample, observations):
    weights = sample["w"]
    predictors = observations["x"]
    return jnp.dot(predictors, weights)


# The combination of prior and distribution form the posterior, from which we
# sampling is performed.

def likelihood(sample, observations):
    sigma = sample["sigma"]
    y = observations["y"]
    y_pred = model(sample, observations)
    return norm.logpdf(y - y_pred, scale=sigma)

def prior(sample):
    return 0.0


# Initial sample
test_w = random.normal(key, (N, 1))

sample = {"w": test_w, "sigma": jnp.array(10.0)}
# Potential strategy
pot_fn = potential.minibatch_potential(prior=prior,
                                      likelihood=likelihood,
                                      strategy="vmap")

# Integrator
integ = integrator.langevin_diffusion(pot_fn, batch_fn)

# Scheduler
step_size = scheduler.polynomial_step_size_first_last(0.005, 0.00001)
sched_fn = scheduler.init_scheduler(step_size)

# Update function

@jax.jit
def update(state, it):
    int_state, sched_state = state
    schedule = sched_fn[2](sched_state)

    int_state = integ[1](int_state, schedule)

    sched_state = sched_fn[1](sched_state)
    return (int_state, sched_state), (integ[2](int_state))

# Inital state
iterations = 10000
sched_init = sched_fn[0](iterations)
int_init = integ[0](sample)

state = (int_init, sched_init)

# for _ in range(iterations):
#   state, _ = update(state, None)

# Run

start = time.time()

@jax.jit
def compute(state):
  _, results = jax.lax.scan(update, state, jnp.arange(iterations))
  return results

init_state = (int_init, sched_init)

print(jax.make_jaxpr(compute)(init_state))
results = compute(init_state)

def numpyro_model(y_obs=None):
  sigma = numpyro.sample("sigma", numpyro.distributions.Uniform(low=0.0, high=10.0))
  weights = numpyro.sample("weights", numpyro.distributions.Uniform(low=-10 * jnp.ones((N, 1)), high=10 * jnp.ones((N, 1))))

  y_pred = jnp.matmul(X, weights)
  numpyro.sample("likelihood", numpyro.distributions.Normal(loc=y_pred, scale=sigma), obs=y_obs)


kernel = numpyro.infer.HMC(numpyro_model)
mcmc = numpyro.infer.MCMC(kernel, 1000, 1000)
mcmc.run(random.PRNGKey(0), y_obs=Y)
mcmc.print_summary()

samples = mcmc.get_samples()

print(w)
print(results["w"][-1,:])
print(results["sigma"][-10:])
print(f"Results in {time.time() - start} seconds")
# print(jax.make_jaxpr(update)((int_init, sched_init), 0.0))

plt.plot(results["sigma"])

plt.figure()
plt.title("w0 vs w1")

plt.scatter(results["w"][-4000:, 0], results["w"][-4000:, 1])
plt.plot(samples["weights"][:, 0], samples["weights"][:, 1], "r")

plt.figure()
plt.title("w2 vs w3")

plt.scatter(results["w"][-4000:, 2], results["w"][-4000:, 3])
plt.plot(samples["weights"][:, 2], samples["weights"][:, 3], "r")

plt.show()