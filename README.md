# Modular stochastic gradient MCMC for Jax

[![CI](https://github.com/tummfm/jax-sgmc/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/tummfm/jax-sgmc/actions/workflows/ci.yml)

**[Introduction](#introduction) | [Content](#content) | [Features](#features) | 
[Quickstart](#quickstart-aliaspy) | [Installation](#installation) |
[Contributing](#contributing)**

## Introduction

## Content

- **[Introduction](#introduction)**
- **[Content](#content)**
- **[Features](#features)**
- **[Quickstart](#quickstart-aliaspy)**
- **[Installation](#installation)** 
- **[Contributing](#contributing)**


## Features

### Data Loading under ``jit``

### Saving PyTrees under ``jit``

## Quickstart ([alias.py](jax_sgmc/alias.py))

A simple example of bayesian linear regression on a toy dataset. It contains all
necessary steps to use a solver from `alias.py`.

#### Setup

````python
import jax                                  
import jax.numpy as jnp                     
from jax import random                      
from jax.scipy.stats import norm            
                                            
from jax_sgmc import data, alias, potential
````

#### Dataset

In this example we first have to generate some data. As the data is already in
the form of jnp-arrays, constructing a `NumpyDataLoader` is the simplest
option. For many datasets such as mnist, using the `TFDataLoader` is recommended,
as it accepts datasets from `tensorflow_datasets`.

```python
N = 4                                                                     
samples = 1000  # Total samples                                           
sigma = 0.5                                                               
                                                                          
key = random.PRNGKey(0)                                                   
split1, split2, split3 = random.split(key, 3)                             
                                                                          
w = random.uniform(split3, minval=-1, maxval=1, shape=(N, 1))             
noise = sigma * random.normal(split2, shape=(samples, 1))                 
x = random.uniform(split1, minval=-10, maxval=10, shape=(samples, N))     
x = jnp.stack([x[:, 0] + x[:, 1], x[:, 1], 0.1 * x[:, 2] - 0.5 * x[:, 3], 
               x[:, 3]]).transpose()                                      
y = jnp.matmul(x, w) + noise                                              
                                                                          
data_loader = data.NumpyDataLoader(x=x, y=y)                              
```

#### Computational Model

The likelihood and prior wrap the model to be investigated. The potential
modules takes care of combining both into a single potential function, which
can be applied to a batch of data or the full dataset.

```python
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
                                                                          
potential_fn = potential.minibatch_potential(prior=prior,                 
                                             likelihood=likelihood,       
                                             strategy='vmap')             
full_potential_fn = potential.full_potential(prior=prior,                 
                                             likelihood=likelihood,       
                                             strategy='vmap')             
```

### SGLD (rms-prop)

SGLD with a polynomial step size schedule and optional speed up via 
RMSprop-adaption.

<https://arxiv.org/abs/1512.07666>

````python

rms_run = alias.sgld(potential_fn,                                      
                     data_loader,                                       
                     cache_size=512,                                    
                     batch_size=10,                                     
                     first_step_size=0.05,                              
                     last_step_size=0.001,                              
                     burn_in=20000,                                     
                     accepted_samples=4000,                             
                     rms_prop=True)                                     
                                                                        
sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(10.0)}             
results = rms_run(sample, iterations=50000)[0]['samples']['variables']  
                                                                                                                        

````

### SGHMC

SGHMC improves the exploratory power of SGLD by introducing momentum.

<https://arxiv.org/abs/1402.4102>

```python
sghmc_run = alias.sghmc(potential_fn,                                  
                        data_loader,                                   
                        cache_size=512,                                
                        batch_size=10,                                 
                        friction=10.0,                                 
                        first_step_size=0.01,                          
                        last_step_size=0.0005,                         
                        burn_in=2000,                                  
                        adapt_noise_model=True,                        
                        diagonal_noise=False)                          
                                                                       
sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}             
results = sghmc_run(sample, iterations=5000)[0]['samples']['variables']
```

### reSGLD

reSGLD simulates a tempered and a default chain in parallel, which exchange 
samples at random following a (biased) markov jump process.

<https://arxiv.org/abs/2008.05367v3>

```python
resgld_run = alias.re_sgld(potential_fn,                                        
                           data_loader,                                         
                           cache_size=512,                                      
                           batch_size=10,                                       
                           first_step_size=0.0001,                              
                           last_step_size=0.000005,                             
                           burn_in=20000,                                       
                           accepted_samples=4000,                               
                           temperature=100.0)                                   
                                                                                
sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}                      
init_samples = [(sample, sample), (sample, sample), (sample, sample)]           
                                                                                
results = resgld_run(*init_samples, iterations=50000)[0]['samples']['variables']
```

### SGGMC

The SGGMC solver is based on the OBABO integrator, which is reversible
when using stochastic gradients. Moreover, the calculation of the full
potential is only necessary once per MH-correction step, which can be applied
after multiple iterations.

<https://arxiv.org/abs/2102.01691>

```python
sggmc_run = alias.sggmc(potential_fn,                                   
                        full_potential_fn,                              
                        data_loader,                                    
                        cache_size=512,                                 
                        batch_size=64,                                  
                        first_step_size=0.005,                          
                        last_step_size=0.0005,                          
                        burn_in=2000)                                   
                                                                        
sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}              
                                                                        
results = sggmc_run(sample, iterations=5000)[0]['samples']['variables'] 
```

### AMAGOLD

The AMAGOLD solver constructs a skew-reversible markov chain, such that
MH-correction steps can be applied periodically to sample from the correct
distribution.

 <https://arxiv.org/abs/2003.00193>

```python
amagold_run = alias.amagold(potential_fn,                                
                            full_potential_fn,                           
                            data_loader,                                 
                            cache_size=512,                              
                            batch_size=64,                               
                            first_step_size=0.005,                       
                            last_step_size=0.0005,                       
                            burn_in=2000)                                
                                                                         
sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}               
                                                                         
results = amagold_run(sample, iterations=5000)[0]['samples']['variables']
```

## Installation

### Jax Installation

Install jax depending on platform (cpu, gpu).

#### For cpu:

```shell
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

#### For gpu:

Please take a look at the 
(jax installation instructions)[https://github.com/google/jax#installation].

### Installation of jax-sgmc

To get the newest version:

```shell
git clone git@github.com:tummfm/jax-sgmc.git
pip install -e ./jax-sgmc
```

### Optional Installs

Some python packages are not essential, but required for some applications of
jax-sgmc:

- TFDataLoader:

```shell
pip install tensorflow tensorflow_datasets
```

- HDF5Collector

```shell
pip install h5py
```

## Contributing

Contributions are very welcome, but take a look at 
[how to contribute](CONTRIBUTING.md).

