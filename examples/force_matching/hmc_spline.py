#%%

import time
from functools import partial

import numpy as onp

import jax
from jax import grad, vmap, tree_util, lax
import jax.numpy as jnp
import jax.scipy as jscp

from jax_md import space, partition
from spline_potential import tabulated_neighbor_list

from jax_sgmc import data, potential, solver, adaption, integrator, scheduler, io
from jax_sgmc.util import host_callback

jax.config.update('jax_enable_x64', True)

#%% md

# Reference Data

#%%

mb_size = 10

forces = onp.load('data/forces_SPC.npy')[0:10000, :, :]
positions = onp.load('data/conf_SPC.npy')[0:10000, :, :]

print(forces.shape)

data_loader = data.NumpyDataLoader(forces=forces, positions=positions)
batch_fn = data.random_reference_data(data_loader, 16, mb_size=mb_size)
data_map = data.full_reference_data(data_loader, 16, mb_size=mb_size)

#%%

# Model

#%%

x_vals = jnp.linspace(0.26, 0.55, num=35)
u_init = jnp.zeros_like(x_vals)

box_size = 3.0
r_cut = 0.9

R_init = tree_util.tree_map(jnp.float32, data_loader.initializer_batch()['positions'])

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
                           dr_threshold=0.2, capacity_multiplier=0.75)


def energy_fn(params, R, **kwargs):
    _neighbor = neighbor_fn(lax.stop_gradient(R), neighbor_init)
    energy = tabulated_energy(params, initialize_neighbor_list=False)
    return energy(R, _neighbor)

vec_energy_fn = vmap(energy_fn, (None, 0))

grad_fn = grad(energy_fn, argnums=1)
force_fn = lambda params, R, **kwargs: -grad_fn(params, R)

its =  0
current_time = time.time()

def log_fun(std, _):
    global its
    its +=  1
    # if onp.mod(its, mb_size) == 0:
    #   print(f"{its / mb_size} iterations in {time.time() - current_time} seconds: {(time.time() - current_time) / its * mb_size} seconds per it, std is {std}")
    print(f"{its } iterations in {time.time() - current_time} seconds: {(time.time() - current_time) / its} seconds per it, std is {std}")


def likelihood(sample, observation):
    sample, observation = tree_util.tree_map(jnp.float32, (sample, observation))
    predicted_forces = force_fn(sample['potential'], observation['positions'])
    # likelihoods = jscp.stats.norm.logpdf(predicted_forces,
    #                                      loc=observation['forces'],
    #                                      scale=sample['std'])
    msq = jnp.sum(jnp.square(predicted_forces - observation['forces']))

    host_callback.id_tap(log_fun, sample['std'])
    return -jnp.float64(msq * jnp.power(sample['std'] * 100, -2.0)) - jnp.log(sample['std']) * onp.product(predicted_forces.shape)

def prior(sample):
    return -jnp.abs(jnp.sum(sample['potential']))

#%% md

# Solver setup

#%%

iterations = 1000

full_potential = potential.full_potential(prior, likelihood, strategy='vmap', full_data_map=data_map[1])
stochastic_potential = potential.minibatch_potential(prior,
                                                     likelihood,
                                                     strategy='vmap')

rms_prop = adaption.rms_prop()
# rms_prop = None

burn_in = scheduler.initial_burn_in(500)
step_size = scheduler.polynomial_step_size_first_last(0.00005, 0.000001)
# step_size = scheduler.adaptive_step_size(1000, initial_step_size=0.0005, speed_constant=0.01, target_acceptance_rate=0.65)


# thinning = scheduler.random_thinning(step_size, burn_in, 300)
sched = scheduler.init_scheduler(step_size=step_size,
                                 burn_in=burn_in)

obabo = integrator.obabo(stochastic_potential, batch_fn, 10, 1500)
# leapfrog = integrator.reversible_leapfrog(stochastic_potential, batch_fn, 10, 0.25)
sgld = solver.sgmc(obabo)
# sggmc = solver.sggmc(obabo, full_potential)
# amagold = solver.amagold(leapfrog, full_potential)

data_collector = io.MemoryCollector(save_dir='hmc_results')
saving = io.save(data_collector)

sgld_mcmc = solver.mcmc(solver=sgld,
                        scheduler=sched,
                        saving=saving)

init_state = sgld[0]({'potential': u_init, 'std': jnp.array(0.5)})

results = sgld_mcmc(init_state, iterations=iterations)

#%%

print(results)
