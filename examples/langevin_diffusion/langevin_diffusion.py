import numpy as onp

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

from numpyro import sample as npy_smpl
import numpyro.infer as npy_inf
import numpyro.distributions as npy_dist

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

from jax_sgmc import adaption
from jax_sgmc import potential
from jax_sgmc import data
from jax_sgmc import scheduler
from jax_sgmc import integrator

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
iterations = 10000

# Adaption strategy
rms_prop = adaption.rms_prop()

# Integrators
default_integrator = integrator.langevin_diffusion(potential_fn,
                                                   batch_fn)
rms_integrator = integrator.langevin_diffusion(potential_fn,
                                               batch_fn,
                                               rms_prop)

# Initial value for starting
sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(10.0)}

# Schedulers
default_step_size = scheduler.polynomial_step_size_first_last(first=0.001,
                                                              last=0.000005)
rms_step_size = scheduler.polynomial_step_size_first_last(first=0.05,
                                                          last=0.001)
default_scheduler = scheduler.init_scheduler(default_step_size)
rms_scheduler = scheduler.init_scheduler(rms_step_size)

# Initial states
default_state = (
default_integrator[0](sample), default_scheduler[0](iterations))
rms_state = (rms_integrator[0](sample), rms_scheduler[0](iterations))


# Update functions
@jax.jit
def default_update(state, _):
  integrator_state, scheduler_state = state

  # Get the schedule for the next step
  schedule = default_scheduler[2](scheduler_state)

  # Integrate
  integrator_state = default_integrator[1](integrator_state, schedule)

  # Update the scheduler
  scheduler_state = default_scheduler[1](scheduler_state)

  return (integrator_state, scheduler_state), default_integrator[2](
    integrator_state)


@jax.jit
def rms_update(state, _):
  integrator_state, scheduler_state = state

  # Get the schedule for the next step
  schedule = rms_scheduler[2](scheduler_state)

  # Integrate
  integrator_state = rms_integrator[1](integrator_state, schedule)

  # Update the scheduler
  scheduler_state = rms_scheduler[1](scheduler_state)

  return (integrator_state, scheduler_state), rms_integrator[2](
    integrator_state)


_, default = jax.lax.scan(default_update, default_state, jnp.arange(iterations))
_, rms = jax.lax.scan(rms_update, rms_state, jnp.arange(iterations))

################################################################################
#
# Results
#
################################################################################


plt.figure()
plt.title("Sigma")

plt.plot(default["sigma"], label="Default")
plt.plot(rms["sigma"], label="RMSprop")

plt.legend()

# Now, only take the last 4000 steps

w_default = default["w"][-4000:, :]
w_rms = rms["w"][-4000:, :]

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

# Default

plt.figure()
plt.title("w_1 vs w_2 (default)")

plt.contour(W1d, W2d, Z12d, levels, colors='red', linewidths=0.5)
plt.plot(w_default[:, 0], w_default[:, 1], 'o', alpha=0.5, markersize=0.5,
         zorder=-1)

# rm


plt.figure()
plt.title("w_1 vs w_2 (rms)")

plt.xlim([0.07, 0.12])
plt.ylim([-0.525, -0.450])
plt.contour(W1d, W2d, Z12d, levels, colors='red', linewidths=0.5)
plt.plot(w_rms[:, 0], w_rms[:, 1], 'o', alpha=0.5, markersize=0.5, zorder=-1)

# w3 vs w4

w34 = gaussian_kde(jnp.squeeze(w_npy[:, 2:4].transpose()))
w3d = onp.linspace(-0.3, -0.05, 100)
w4d = onp.linspace(-0.75, -0.575, 100)
W3d, W4d = onp.meshgrid(w3d, w4d)
p34d = onp.vstack([W3d.ravel(), W4d.ravel()])
Z34d = onp.reshape(w34(p34d).T, W3d.shape)
Z34d /= Z34d.max()

# Default

plt.figure()

plt.title("w_3 vs w_4 (default)")

plt.contour(W3d, W4d, Z34d, levels, colors='red', linewidths=0.5)
plt.plot(w_default[:, 2], w_default[:, 3], 'o', alpha=0.5, markersize=0.5,
         zorder=-1)

# rm


plt.figure()
plt.title("w_3 vs w_4 (rms)")
plt.contour(W3d, W4d, Z34d, levels, colors='red', linewidths=0.5)
plt.plot(w_rms[:, 2], w_rms[:, 3], 'o', alpha=0.5, markersize=0.5, zorder=-1)

plt.show()
