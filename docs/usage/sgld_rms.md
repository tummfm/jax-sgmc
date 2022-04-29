---
jupytext:
  formats: ipynb,md:myst,py
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
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

```{code-cell}
:tags: [hide-cell]
import numpy as onp

import jax.numpy as jnp

from jax import random

import matplotlib.pyplot as plt

from jax.scipy.stats import norm

from numpyro import sample as npy_smpl
import numpyro.infer as npy_inf
import numpyro.distributions as npy_dist

from scipy.stats import gaussian_kde


from jax_sgmc import potential
from jax_sgmc.data.numpy_loader import NumpyDataLoader
from jax_sgmc import alias
```

# Setup Solver

## Structure of SGLD Solver

```{mermaid}
flowchart TD
%%{config: {'height': 'auto'}}%%

  MCMC((MCMC))

  subgraph B [Scheduler]
    direction TB
    B1[Update] --> B11[Update Specific Schedulers]
    B2[Get] --> B21[Get Specific Schedules]
  end

  subgraph D [Solver]
    direction TB
    D1[Update Solver State]
    D2[Get Samples from Solver State]
  end

  subgraph C [Integrator]
    direction TB
    CU[Update Integrator State]
    CG[Get Variables from Integrator State]
  end

  subgraph E [Adaption]
    direction TB
    E1[Update Adaption State]
    E2[Get Manifold from Adaption State]
  end

  subgraph F [Data]
    direction TB
    F1[Draw Random Batch] --> F2{Cache Empty?}
    F2 -- No --> F3[Load Cached Batch]
  end
  
  subgraph FDL [Data Loader]
    direction TB
    F2 -. Yes .-> FDL1[Assemble New Cache] -.-> F3
  end

  subgraph H [Potential]
    direction TB
    H1[Evaluate Stochastic Potential]
  end
  
  subgraph I [Saving]
    direction TB
    I1[Save] --> I2{Thinning / Burn In?}
    I2 --> I3[Update Statistics]
  end

  subgraph J [Data Collector]
    J1[Save Sample]
  end

  I2 -. Yes .-> J1

  MCMC --> B1
  MCMC --> B2
  MCMC --> D1
  MCMC --> D2
  MCMC --> I1
  
  D1 ---> CU
  D2 ---> CG
  
  H1 -->  U1{{Evaluate User Likelihood}}
  H1 --> U2{{Evaluate User Prior}}

  CU --> E1
  CU --> E2
  CU --> F1
  CU --> H1

```

## Construct Solver

The Solver is applied to the problem in quickstart.


### Setup Reference Data Loading

The reference data is passed to the solver via two components, the data loader
and the host callback wrapper.

The host callback wrappers load the data into the jit-compiled programs via
``host_callback.call()``. To balance the memory usage and the delay of loading
the data, a number of batches is loaded in each call.

The data loader assembles the batches requested by the host callback wrappers.
It loads the data from a source (HDF-File, numpy-array, tensorflow dataset)
and selects the observations in each batch after a specific method
(ordered access, shuffling). Which of those methods are available differe
between the data loaders.


```{code-cell}
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

```{code-cell}

# The construction of the data loader can be different. For the numpy data
# loader, the numpy arrays can be passed as keyword arguments and are later
# returned as a dictionary with corresponding keys.
data_loader = NumpyDataLoader(x=x, y=y)

# The cache size corresponds to the number of batches per cache. The state
# initialized via the init function is necessary to identify which data chain
# request new batches of data.
init_fn, batch_fn = data.random_reference_data(data_loader,
                                               mb_size=N,
                                               cache_size=100)

```

### Setup Potential

The model is connected to the solver via the (log-)prior and (log-)likelihood
function. The model for our problem is:

```{code-cell}
def model(sample, observations):
    weights = sample["w"]
    predictors = observations["x"]
    return jnp.dot(predictors, weights)

```
**JaxSGMC** supports samples in the form of pytrees, so no flattering of e.g.
Neural Net parameters is necessary. In our case we can separate the standard
deviation, which is only part of the likelihood, from the weights by using a
dictionary:

```{code-cell}

def likelihood(sample, observations):
    sigma = sample["sigma"]
    y = observations["y"]
    y_pred = model(sample, observations)
    return norm.logpdf(y - y_pred, scale=sigma)

def prior(sample):
    del sample
    return 0.0
    
```

The prior and likelihood are not passed to the solver directly, but 
first transformed into a (stochastic) potential.
This allowed us to formulate the model and so the likelihood with only a single 
observation in mind and let **JaxSGMC** take care of evaluating it for a batch
of observations. As the model is not computationally demanding, we let 
**JaxSGMC** vectorize the evaluation of the likelihood:

```{code-cell}

potential_fn = potential.minibatch_potential(prior=prior,
                                             likelihood=likelihood,
                                             strategy="vmap")                                    
```

### Setup Adaption

### Setup Integrator and Solver

### Setup Scheduler

### Setup Saving

### Run Solver