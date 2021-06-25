# Copyright 2021 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as onp

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
from jax import tree_util

from jax_sgmc import adaption
from jax_sgmc import potential
from jax_sgmc import data
from jax_sgmc import io
from jax_sgmc import scheduler
from jax_sgmc import integrator
from jax_sgmc import solver

import time
################################################################################
#
# Reference Data
#
################################################################################

N = 4
samples = 1000  # Total samples
sigma = 0.5

key = random.PRNGKey(0)
split1, split2, split3 = random.split(key, 3)

w = random.uniform(split3, minval=-1, maxval=1, shape=(N, 1))
noise = jnp.sqrt(sigma) * random.normal(split2, shape=(samples, 1))
x = random.uniform(split1, minval=-10, maxval=10, shape=(samples, N))
x = jnp.stack([x[:, 0] + x[:, 1], x[:, 1], 0.1 * x[:, 2] - 0.5 * x[:, 3],
               x[:, 3]]).transpose()
y = jnp.matmul(x, w) + noise


################################################################################
#
# Solution with Jax SGMC
#
################################################################################


M = 10
cs = 1000

data_loader = data.NumpyDataLoader(M, x=x, y=y)
batch_fn = data.random_reference_data(data_loader, cached_batches_count=cs)


# == Model definition ==========================================================

def model(sample, observations):
  weights = sample["w"]
  predictors = observations["x"]
  return jnp.dot(predictors, weights)


def likelihood(sample, observations):
  sigma = sample["sigma"]
  y = observations["y"]
  y_pred = model(sample, observations)
  return norm.logpdf(y - y_pred, scale=sigma)


def prior(unused_sample):
  return 0.0


# If the model is more complex, the strategy can be set to map for sequential
# evaluation and pmap for parallel evaluation.
potential_fn = potential.minibatch_potential(prior=prior,
                                             likelihood=likelihood,
                                             strategy="vmap")

# == Solver Setup ==============================================================

# Number of iterations
iterations = 50000

# Integrators
default_integrator = integrator.langevin_diffusion(potential_fn,
                                                   batch_fn)

# Initial value for starting
sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(10.0)}

# Schedulers
default_step_size = scheduler.polynomial_step_size_first_last(first=0.001,
                                                              last=0.000005)

burn_in = scheduler.initial_burn_in(20000)
default_random_thinning = scheduler.random_thinning(default_step_size, burn_in, 200)

default_scheduler = scheduler.init_scheduler(step_size=default_step_size,
                                             burn_in=burn_in,
                                             thinning=default_random_thinning)

default_sgld = solver.sgmc(default_integrator)

data_collector = io.JSONCollector("json", write_frequency=50)
saving = io.save(data_collector=data_collector)

default_run = solver.mcmc(default_sgld, default_scheduler, saving=saving)

#default = default_run(default_integrator[0](sample), iterations=iterations)

init_state = default_integrator[0](sample)

import h5py

def serialize(state):
  flat_state, tree_def = tree_util.tree_flatten(state)
  data = {f"table~{idx}": table for idx, table in enumerate(flat_state)}
  return data

file = h5py.File("hdf5/checkpoint.hdf5", "a")
for key, value in serialize(init_state).items():
  file.create_dataset(name=key, data=value)

file.close()

print(init_state)
print(serialize(init_state))


