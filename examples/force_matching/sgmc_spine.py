#%%

from functools import partial

import numpy as onp

from jax import grad, vmap
import jax.numpy as jnp
import jax.scipy as jscp

from jax_md import space, partition
from spline_potential import tabulated_neighbor_list

from jax_sgmc import data, potential, solver, adaption, integrator, scheduler, io
from jax_sgmc.util import host_callback

#%% md

# Reference Data

#%%

mb_size = 4

forces = onp.load('data/forces_SPC.npy')
positions = onp.load('data/conf_SPC.npy')

data_loader = data.NumpyDataLoader(forces=forces, positions=positions)
batch_fn = data.random_reference_data(data_loader, 256, mb_size=mb_size)

#%%

# Model

#%%

x_vals = jnp.linspace(0.26, 0.65, num=25)
u_init = jnp.zeros_like(x_vals)

box_size = 3.0
r_cut = 0.9

R_init = data_loader.initializer_batch()['positions']

print(R_init)

displacement, shift = space.periodic(box_size)
displacement_fn, shift_fn = space.periodic(box_size)

neighbor_fn = partition.neighbor_list(displacement_fn,
                                      box_size,
                                      r_cut,
                                      dr_threshold=0.05,
                                      capacity_multiplier=1.5)
neighbor_init = neighbor_fn(R_init, extra_capacity=0)

tabulated_energy = partial(tabulated_neighbor_list, displacement, x_vals,
                           box_size=box_size, r_onset=(r_cut - 0.2), r_cutoff=r_cut,
                           dr_threshold=0.2, capacity_multiplier=1.0)


def energy_fn(params, R, **kwargs):
    _neighbor = neighbor_fn(R, neighbor_init)
    energy = tabulated_energy(params, initialize_neighbor_list=False)
    return energy(R, _neighbor)

vec_energy_fn = vmap(energy_fn, (None, 0))

grad_fn = grad(energy_fn, argnums=1)
grad_energy_lambda_fn = grad(energy_fn, argnums=0)
force_fn = lambda params, R, **kwargs: -grad_fn(params, R)

def likelihood(sample, observation):
    predicted_forces = force_fn(sample['potential'], observation['positions'])
    likelihoods = jscp.stats.norm.logpdf(predicted_forces,
                                         loc=observation['forces'],
                                         scale=sample['std'])
    host_callback.id_print(sample['std'], what="std")
    return jnp.sum(likelihoods)

def prior(sample):
    likelihoods = jscp.stats.norm.logpdf(sample['potential'],
                                         loc=jnp.zeros_like(sample['potential']),
                                         scale=sample['std'])
    return jnp.sum(likelihoods)

#%% md

# Solver setup

#%%

iterations = 100000

stochastic_potential = potential.minibatch_potential(prior,
                                                     likelihood,
                                                     strategy='vmap')

#rms_prop = adaption.rms_prop()
rms_prop = None

burn_in = scheduler.initial_burn_in(50000)
step_size = scheduler.polynomial_step_size_first_last(0.1, 0.05)
thinning = scheduler.random_thinning(step_size, burn_in, 300)
sched = scheduler.init_scheduler(step_size=step_size,
                                 burn_in=burn_in,
                                 thinning=thinning)

ld = integrator.langevin_diffusion(stochastic_potential,
                                           batch_fn,
                                           rms_prop)
sgld = solver.sgmc(ld)

data_collector = io.MemoryCollector()
saving = io.save(data_collector)

sgld_mcmc = solver.mcmc(solver=sgld,
                        scheduler=sched,
                        saving=saving)

init_state = sgld[0]({'potential': u_init, 'std': jnp.array(200.0)})

results = sgld_mcmc(init_state, iterations=iterations)

#%%

print(results)
