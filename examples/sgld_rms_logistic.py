import h5py
import matplotlib.pyplot as plt
import numpy as onp
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

from jax_sgmc import data, potential, scheduler, adaption, integrator, solver, io
from jax_sgmc.data.numpy_loader import NumpyDataLoader

N = 4
samples = 1000  # Total samples

key = random.PRNGKey(0)
split1, split2, split3, split4 = random.split(key, 4)

# Correct solution
w = random.uniform(split3, minval=-1, maxval=1, shape=(N, 1))

# Data generation
x = random.randint(split1, minval=-10, maxval=10, shape=(samples, N))
x = jnp.stack([x[:, 0] + x[:, 1], x[:, 1], 0.1 * x[:, 2] - 0.5 * x[:, 3],
               x[:, 3]]).transpose()
y = random.bernoulli(split4, 1 / (1 + jnp.exp(-jnp.matmul(x, w))))
y = y.astype(float)
data_loader = NumpyDataLoader(x=x, y=y)

data_fn = data.random_reference_data(data_loader,
                                     mb_size=N,
                                     cached_batches_count=100)

data_loader_kwargs = {
    "seed": 0,
    "shuffle": True,
    "in_epochs": False
}


def model(sample, observations):
    weights = sample["w"]
    predictors = observations["x"]
    return 1 / (1 + jnp.exp(-jnp.matmul(predictors, weights)))


def likelihood(sample, observations):
    sigma = sample["log_sigma"]
    y = observations["y"]
    y_pred = model(sample, observations)
    return (y * jnp.log(y_pred / (1 - y_pred)) + jnp.log(1 - y_pred))


def prior(sample):
    params = sample["w"]
    log_pdf = norm.logpdf(params, loc=0., scale=10)
    return log_pdf.sum()

potential_fn = potential.minibatch_potential(prior=prior,
                                             likelihood=likelihood,
                                             strategy="vmap")
rms_prop_adaption = adaption.rms_prop()

adaption_kwargs = {
    "lmbd": 1e-6,
    "alpha": 0.99
}

langevin_diffusion = integrator.langevin_diffusion(potential_fn=potential_fn,
                                                   batch_fn=data_fn,
                                                   adaption=rms_prop_adaption)

rms_prop_solver = solver.sgmc(langevin_diffusion)

init_sample = {"log_sigma": jnp.array(0.0), "w": jnp.zeros(N)}
init_state = rms_prop_solver[0](init_sample,
                                adaption_kwargs=adaption_kwargs,
                                batch_kwargs=data_loader_kwargs)
step_size_schedule = scheduler.polynomial_step_size_first_last(first=0.001,
                                                               last=0.0001)
burn_in_schedule = scheduler.initial_burn_in(10000)
thinning_schedule = scheduler.random_thinning(step_size_schedule=step_size_schedule,
                                              burn_in_schedule=burn_in_schedule,
                                              selections=10000)

schedule = scheduler.init_scheduler(step_size=step_size_schedule,
                                    burn_in=burn_in_schedule,
                                    thinning=thinning_schedule)

data_collector = io.MemoryCollector()
save_fn = io.save(data_collector=data_collector)

mcmc = solver.mcmc(solver=rms_prop_solver,
                   scheduler=schedule,
                   saving=save_fn)

# Take the result of the first chain
results = mcmc(init_state, iterations=50000)[0]

print(f"Collected {results['sample_count']} samples")

w_rms = results["samples"]["variables"]["w"]

for i in range(N):
    fig, ax = plt.subplots(1, 1)
    ax.hist(w_rms[:, i])
    plt.title("This is for weight " + str(i))
    plt.show()
    print("mean of weight=" + str(i) + " is " + str(jnp.mean(w_rms[:, i])) + ", w_true=" + str(w[i]))
