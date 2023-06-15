---
jupytext:
  formats: examples///ipynb,examples///md:myst,docs//usage//ipynb
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3
  name: python3
---

```{raw-cell}

---
Copyright 2021 Multiscale Modeling of Fluid Materials, TU Munich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
---
```

```{code-cell} ipython3
:tags: [hide-cell]

import warnings
warnings.filterwarnings('ignore')

import h5py
import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

from jax_sgmc import data, potential, scheduler, adaption, integrator, solver, io
from jax_sgmc.data.numpy_loader import NumpyDataLoader
from jax_sgmc.data.hdf5_loader import HDF5Loader
```

# Setup Custom Solver

This example shows how to customize a solver by combining the individual
modules of **JaxSGMC**.
It covers all necessary steps to build a *SGLD* solver with *RMSprop* adaption
and applies it to the same problem as described in {doc}`/quickstart/`.

## Overview

Schematically, a solver in **JaxSGMC** has the structure:

![Structure of JaxSGMC](https://raw.githubusercontent.com/tummfm/jax-sgmc/main/jax-sgmc-structure.svg)

The SGLD solver with RMSprop adaption will make use of all modules.
It is set up in these steps:

- **[Load Reference Data](#load-reference-data)**
- **[Transform Log-likelihood to Potential](#transform-log-likelihood-to-potential)**
- **[RMSprop Adaption](#rmsprop-adaption)**
- **[Integrator and Solver](#integrator-and-solver)**
- **[Scheduler](#scheduler)**
- **[Save Samples](#save-samples-in-numpy-arrays)**
- **[Run Solver](#run-solver)**

## Load Reference Data

The reference data is passed to the solver via two components, the Data Loader
and the Host Callback Wrapper.

The Data Loader assembles the batches requested by the host callback wrappers.
It loads the data from a source (HDF-File, numpy-array, tensorflow dataset)
and selects the observations in each batch after a specific method
(ordered access, shuffling, ...).

The Host Callback Wrapper requests new batches from the Data Loader and loads
them into jit-compiled programs via Jax's Host Callback module.
To balance the memory usage and the delay due to loading the data, each device
call returns multiple batches.

```{code-cell} ipython3
:tags: [hide-cell]

N = 4 
samples = 1000 # Total samples

key = random.PRNGKey(0)
split1, split2, split3 = random.split(key, 3)

# Correct solution
sigma = 0.5
w = random.uniform(split3, minval=-1, maxval=1, shape=(N, 1))

# Data generation
noise = sigma * random.normal(split2, shape=(samples, 1))
x = random.uniform(split1, minval=-10, maxval=10, shape=(samples, N))
x = jnp.stack([x[:, 0] + x[:, 1], x[:, 1], 0.1 * x[:, 2] - 0.5 * x[:, 3],
               x[:, 3]]).transpose()
y = jnp.matmul(x, w) + noise
```

The NumpyDataLoader assembles batches randomly by drawing from the the complete
dataset with and without replacement (shuffling).
It also provides the possibility to start the batching from a defined state, 
controlled via the seed.

These settings can be passed differently for every chain and are thus not passed
during the initialization.
Instead, they have to be passed during the
[initialization of the chains](#integrator-and-solver).

In this example, the batches are shuffled, i.e. every sample is used at least
once before an already drawn sample is used again and the chains start at a
defined state.

```{code-cell} ipython3
# The construction of the data loader can be different. For the numpy data
# loader, the numpy arrays can be passed as keyword arguments and are later
# returned as a dictionary with corresponding keys.
data_loader = NumpyDataLoader(x=x, y=y)

# The cache size corresponds to the number of batches per cache. The state
# initialized via the init function is necessary to identify which data chain
# request new batches of data.
data_fn = data.random_reference_data(data_loader,
                                     mb_size=N,
                                     cached_batches_count=100)

data_loader_kwargs = {
    "seed": 0,
    "shuffle": True,
    "in_epochs": False
}
```

## Transform Log-likelihood to Potential

The model is connected to the solver via the (log-)prior and (log-)likelihood
function. The model for our problem is:

```{code-cell} ipython3
def model(sample, observations):
    weights = sample["w"]
    predictors = observations["x"]
    return jnp.dot(predictors, weights)
```

**JaxSGMC** supports samples in the form of pytrees, so no flattering of e.g.
neural network parameters is necessary. In our case we can separate the standard
deviation, which is only part of the likelihood, from the weights by using a
dictionary:

```{code-cell} ipython3
sample = {"log_sigma": jnp.array(1.0), "w": jnp.zeros((N, 1))}

def likelihood(sample, observations):
    sigma = jnp.exp(sample["log_sigma"])
    y = observations["y"]
    y_pred = model(sample, observations)
    return norm.logpdf(y - y_pred, scale=sigma)

def prior(sample):
    return 1 / jnp.exp(sample["log_sigma"])
    
```

The prior and likelihood are not passed to the solver directly, but 
first transformed into a (stochastic) potential.
This allows us to formulate the model and so the likelihood with only a single 
observation in mind and let **JaxSGMC** take care of evaluating it for a batch
of observations. As the model is not computationally demanding, we let 
**JaxSGMC** vectorize the evaluation of the likelihood:

```{code-cell} ipython3
potential_fn = potential.minibatch_potential(prior=prior,
                                             likelihood=likelihood,
                                             strategy="vmap")                                    
```

## RMSprop Adaption

The adaption module simplifies the implementation of an adaption strategy
by raveling / unraveling the latent variables pytree.

The RMSprop adaption is characterized by two parameters, which can be set
dynamically for each chain.
As for the data loader arguments, non-default RMSprop parameters must be passed
during the [initialization of the chains](#integrator-and-solver).

```{code-cell} ipython3
rms_prop_adaption = adaption.rms_prop()

adaption_kwargs = {
    "lmbd": 1e-6,
    "alpha": 0.99
}
```

## Integrator and Solver

The integrator proposes new samples based on a specific process which are then
processed by the solver.
For example, the solver might reject a proposal by a Metropolis Hastings
acceptance step (AMAGOLD, SGGMC) or swap it with another proposal by a parallel
tempering chain swap (reSGLD).

In this case, a Langevin Diffusion process proposes a new sample, which is
accepted unconditionally by the solver.

After this step we defined our process.
Therefore, we can now initialize the starting states of each chain with the
dynamic settings for the data loader and adaption.

```{code-cell} ipython3
langevin_diffusion = integrator.langevin_diffusion(potential_fn=potential_fn,
                                                   batch_fn=data_fn,
                                                   adaption=rms_prop_adaption)

# Returns a triplet of init_fn, update_fn and get_fn
rms_prop_solver = solver.sgmc(langevin_diffusion)

# Initialize the solver by providing initial values for the latent variables.
# We provide extra arguments for the data loader and the adaption method.
init_sample = {"log_sigma": jnp.array(0.0), "w": jnp.zeros(N)}
init_state = rms_prop_solver[0](init_sample,
                                adaption_kwargs=adaption_kwargs,
                                batch_kwargs=data_loader_kwargs)
```

## Scheduler

Next, we set up a schedule which updates process parameters such as the
temperature and the step size independently of the solver state.
It is moreover necessary to determine which samples should be saved or discarded.

SGLD only depends on the step size, which is chosen to follow a polynomial
schedule.
However, as only a few and independent samples should be saved, we also set up a
burn in schedule, which rejects the first 2000 samples and a thinning schedule,
which randomly selects 1000 samples not subject to burn in.

```{code-cell} ipython3
step_size_schedule = scheduler.polynomial_step_size_first_last(first=0.05,
                                                               last=0.001,
                                                               gamma=0.33)
burn_in_schedule = scheduler.initial_burn_in(2000)
thinning_schedule = scheduler.random_thinning(step_size_schedule=step_size_schedule,
                                              burn_in_schedule=burn_in_schedule,
                                              selections=1000)

# Bundles all specific schedules
schedule = scheduler.init_scheduler(step_size=step_size_schedule,
                                    burn_in=burn_in_schedule,
                                    thinning=thinning_schedule)
```

## Save samples in numpy Arrays

By default, **JaxSGMC** save accepted samples in the device memory.
However, for some models the required memory rapidly exceeds the available 
memory. Therefore, **JaxSGMC** supports saving the samples on the host in a
similar manner as it loads reference data from the host.

Hence, also the saving step consists of setting up a Data Collector, which takes
care of saving the data in different formats and a general Host Callback Wrapper
which transfers the data out of jit-compiled computations.

In this example, the data is simply passed to (real) numpy arrays in the host
memory.

```{code-cell} ipython3
data_collector = io.MemoryCollector()
save_fn = io.save(data_collector=data_collector)
```

### Save samples in hdf5

```{code-cell} ipython3
import h5py

data_collector = io.MemoryCollector()
save_fn = io.save(data_collector=data_collector)
```

## Run Solver

Finally, all parts of the solver are set up and can be combined to a runnable
process.
The mcmc function updates the scheduler and integrator in the correct order and
passes the results to the saving module. 

The mcmc function can be called with multiple ``init_states`` as
positional arguments to run multiple chains and returns a list of results, one
for each chain.

```{code-cell} ipython3
mcmc = solver.mcmc(solver=rms_prop_solver,
                   scheduler=schedule,
                   saving=save_fn)

# Take the result of the first chain
results = mcmc(init_state, iterations=10000)[0]


print(f"Collected {results['sample_count']} samples")
```

## Plot Results

```{code-cell} ipython3
:tags: [hide-input]

plt.figure()
plt.title("Sigma")

plt.plot(onp.exp(results["samples"]["variables"]["log_sigma"]), label="RMSprop")

w_rms = results["samples"]["variables"]["w"]

# w1 vs w2
w1d = onp.linspace(0.00, 0.20, 100)
w2d = onp.linspace(-0.70, -0.30, 100)
W1d, W2d = onp.meshgrid(w1d, w2d)
p12d = onp.vstack([W1d.ravel(), W2d.ravel()])

plt.figure()
plt.title("w_1 vs w_2 (rms)")

plt.xlim([0.07, 0.12])
plt.ylim([-0.525, -0.450])
plt.plot(w_rms[:, 0], w_rms[:, 1], 'o', alpha=0.5, markersize=0.5, zorder=-1)

# w3 vs w4
w3d = onp.linspace(-0.3, -0.05, 100)
w4d = onp.linspace(-0.75, -0.575, 100)
W3d, W4d = onp.meshgrid(w3d, w4d)
p34d = onp.vstack([W3d.ravel(), W4d.ravel()])

plt.figure()
plt.title("w_3 vs w_4 (rms)")
plt.plot(w_rms[:, 2], w_rms[:, 3], 'o', alpha=0.5, markersize=0.5, zorder=-1)
```

## Large Models: Save Data to HDF5

```{code-cell} ipython3
# Open a HDF5 file to store data in
with h5py.File("sgld_rms.hdf5", "w") as file:

    data_collector = io.HDF5Collector(file)
    save_fn = io.save(data_collector=data_collector)

    mcmc = solver.mcmc(solver=rms_prop_solver,
                   scheduler=schedule,
                   saving=save_fn)

    # The solver has to be reinitialized, as the data loader has to be reinitialized
    init_state = rms_prop_solver[0](init_sample,
                                    adaption_kwargs=adaption_kwargs,
                                    batch_kwargs=data_loader_kwargs)
    results = mcmc(init_state, iterations=10000)[0]

print(f"Collected {results['sample_count']} samples")
```

```{code-cell} ipython3
# Sum up and count all values
def map_fn(batch, mask, count):
    return jnp.sum(batch["w"].T * mask, axis=1), count + jnp.sum(mask)

# Load only the samples from the file
with h5py.File("sgld_rms.hdf5", "r") as file:
    postprocess_loader = HDF5Loader(file, subdir="/chain~0/variables", sample=init_sample)

    full_data_mapper, _ = data.full_data_mapper(postprocess_loader, 128, 128)
    w_sums, count = full_data_mapper(map_fn, 0, masking=True)

    # Sum up the sums from the individual batches
    w_means = jnp.sum(w_sums, axis=0) / count

print(f"Collected {count} samples with means:")
for idx, (w, w_old) in enumerate(zip(w_means, onp.mean(w_rms, axis=0))):
  print(f"  w_{idx}: new = {w}, old = {w_old})")
```

```{code-cell} ipython3
:tags: [hide-cell]

import os
os.remove("sgld_rms.hdf5")
```
