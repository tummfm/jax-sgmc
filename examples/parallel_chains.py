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

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

from jax import test_util
test_util.set_host_platform_device_count(4)

from numpyro import sample as npy_smpl
import numpyro.infer as npy_inf
import numpyro.distributions as npy_dist

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

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
# Solution with numpyro
#
################################################################################


def numpyro_model(y_obs=None):
  sigma = npy_smpl("sigma", npy_dist.Uniform(low=0.0, high=10.0))
  weights = npy_smpl("weights",
                     npy_dist.Uniform(low=-10 * jnp.ones((N, 1)),
                                      high=10 * jnp.ones((N, 1))))

  y_pred = jnp.matmul(x, weights)
  npy_smpl("likelihood", npy_dist.Normal(loc=y_pred, scale=sigma), obs=y_obs)


# Collect 1000 samples

kernel = npy_inf.HMC(numpyro_model)
mcmc = npy_inf.MCMC(kernel, 1000, 1000)
mcmc.run(random.PRNGKey(0), y_obs=y)
mcmc.print_summary()

w_npy = mcmc.get_samples()["weights"]

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

# Adaption strategy
rms_prop = adaption.rms_prop()

# Integrators
default_integrator = integrator.langevin_diffusion(potential_fn,
                                                   batch_fn)
rms_integrator = integrator.langevin_diffusion(potential_fn,
                                               batch_fn,
                                               rms_prop)





# Schedulers

rms_step_size = scheduler.polynomial_step_size_first_last(first=0.05,
                                                          last=0.001)
burn_in = scheduler.initial_burn_in(20000)
rms_random_thinning = scheduler.random_thinning(rms_step_size, burn_in, 100)


rms_scheduler = scheduler.init_scheduler(step_size=rms_step_size,
                                         burn_in=burn_in,
                                         thinning=rms_random_thinning)

rms_sgld = solver.sgmc(rms_integrator)

# Data collector
data_collector = io.MemoryCollector()
saving = io.save(data_collector)

rms_run = solver.mcmc(rms_sgld, rms_scheduler, strategy="vmap", saving=saving)

# Initial value for starting
init_sample = lambda seed: {"w": 2 * random.normal(random.PRNGKey(0), (N, 1)),
                            "sigma": jnp.array(10.0)}
init_states = list(map(rms_integrator[0], (init_sample(seed) for seed in range(15))))


rms_results = rms_run(*init_states, iterations=iterations)

# Append the results of the chains
rms_likelihoods = onp.array(jnp.squeeze(jnp.concatenate([rms["samples"]["likelihood"] for rms in rms_results], axis=0)))
w_rms = onp.array(jnp.squeeze(jnp.concatenate([rms["samples"]["variables"]["w"] for rms in rms_results], axis=0)))
sigma_rms = onp.array(jnp.squeeze(jnp.concatenate([rms["samples"]["variables"]["sigma"] for rms in rms_results], axis=0)))

################################################################################
#
# Results
#
################################################################################
print(rms_results[0]["samples"]["variables"]["sigma"][0:10])
print(rms_results[1]["samples"]["variables"]["sigma"][0:10])

plt.figure()
plt.title("Sigma")

plt.plot(sigma_rms, label="RMSprop")

plt.legend()


# Contours of numpyro solution

levels = onp.linspace(0.1, 1.0, 5)

# w1 vs w2
w12 = gaussian_kde(jnp.squeeze(w_npy[:, 0:2].transpose()))
w1d = onp.linspace(0.00, 0.20, 100)
w2d = onp.linspace(-0.70, -0.30, 100)
W1d, W2d = onp.meshgrid(w1d, w2d)
p12d = onp.vstack([W1d.ravel(), W2d.ravel()])
Z12d = onp.reshape(w12(p12d).T, W1d.shape)
Z12d /= Z12d.max()

# rm


plt.figure()
plt.title("w_1 vs w_2 (rms)")

plt.xlim([0.07, 0.12])
plt.ylim([-0.525, -0.450])
plt.contour(W1d, W2d, Z12d, levels, colors='red', linewidths=0.5)
plt.scatter(w_rms[:, 0], w_rms[:, 1], marker='o', c=rms_likelihoods, s=0.3, zorder=-1)

# w3 vs w4

w34 = gaussian_kde(jnp.squeeze(w_npy[:, 2:4].transpose()))
w3d = onp.linspace(-0.3, -0.05, 100)
w4d = onp.linspace(-0.75, -0.575, 100)
W3d, W4d = onp.meshgrid(w3d, w4d)
p34d = onp.vstack([W3d.ravel(), W4d.ravel()])
Z34d = onp.reshape(w34(p34d).T, W3d.shape)
Z34d /= Z34d.max()


# rm

plt.figure()
plt.title("w_3 vs w_4 (rms)")
plt.contour(W3d, W4d, Z34d, levels, colors='red', linewidths=0.5)
plt.scatter(w_rms[:, 2], w_rms[:, 3], marker='o', c=rms_likelihoods, s=0.3, zorder=-1)
plt.colorbar()

plt.show()
