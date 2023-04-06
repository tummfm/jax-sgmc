import h5py
import matplotlib.pyplot as plt
import numpy as onp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str('3')  # needs to stay before importing jax
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

from jax_sgmc import data, potential, scheduler, adaption, integrator, solver, io
from jax_sgmc.data.numpy_loader import NumpyDataLoader
from jax_sgmc.data.hdf5_loader import HDF5Loader
import optax


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
N = 4
samples = 1000 # Total samples

key = random.PRNGKey(0)
split1, split2, split3, split4 = random.split(key, 4)

# Correct solution
#sigma = 0.5
w = random.uniform(split3, minval=-1, maxval=1, shape=(N, 1))

# Data generation
# noise = sigma * random.normal(split2, shape=(samples, 1))
x = random.randint(split1, minval=-10, maxval=10, shape=(samples, N))
x = jnp.stack([x[:, 0] + x[:, 1], x[:, 1], 0.1 * x[:, 2] - 0.5 * x[:, 3],
               x[:, 3]]).transpose()
# y = jnp.matmul(x, w) + noise
y = random.bernoulli(split4, 1/(1+jnp.exp(-jnp.matmul(x,w))))
y = y.astype(float)
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

def model(sample, observations):
    weights = sample["w"]
    predictors = observations["x"]
    return 1/(1+jnp.exp(-jnp.matmul(predictors, weights)))

def likelihood(sample, observations):
    sigma = sample["log_sigma"]
    y = observations["y"]
    y_pred = model(sample, observations)
    # return optax.sigmoid_binary_cross_entropy(y, y_pred)
    return (y*jnp.log(y_pred/(1-y_pred))+jnp.log(1-y_pred))

def prior(sample):
    params = sample["w"]
    log_pdf = norm.logpdf(params, loc = 0., scale = 10)
    return log_pdf.sum()

# def prior(sample):
#     # return random.uniform(split3, jnp.shape(sample["log_sigma"]), minval=-10, maxval=10)#1 / jnp.exp(sample["log_sigma"])
#     return 1 # 0 * random.normal(split2, shape=(1,))

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

# Returns a triplet of init_fn, update_fn and get_fn
rms_prop_solver = solver.sgmc(langevin_diffusion)

# Initialize the solver by providing initial values for the latent variables.
# We provide extra arguments for the data loader and the adaption method.
init_sample = {"log_sigma": jnp.array(0.0), "w": jnp.zeros(N)}
init_state = rms_prop_solver[0](init_sample,
                                adaption_kwargs=adaption_kwargs,
                                batch_kwargs=data_loader_kwargs)
step_size_schedule = scheduler.polynomial_step_size_first_last(first=0.001,
                                                               last=0.0001)
burn_in_schedule = scheduler.initial_burn_in(500000)
thinning_schedule = scheduler.random_thinning(step_size_schedule=step_size_schedule,
                                              burn_in_schedule=burn_in_schedule,
                                              selections=50000)

# Bundles all specific schedules
schedule = scheduler.init_scheduler(step_size=step_size_schedule,
                                    burn_in=burn_in_schedule,
                                    thinning=thinning_schedule)

data_collector = io.MemoryCollector()
save_fn = io.save(data_collector=data_collector)

import h5py

data_collector = io.MemoryCollector()
save_fn = io.save(data_collector=data_collector)

mcmc = solver.mcmc(solver=rms_prop_solver,
                   scheduler=schedule,
                   saving=save_fn)

# Take the result of the first chain
results = mcmc(init_state, iterations=1000000)[0]


print(f"Collected {results['sample_count']} samples")

# plt.figure()
# plt.title("Sigma")
#
# plt.plot(onp.exp(results["samples"]["variables"]["log_sigma"]), label="RMSprop")
# plt.show()
w_rms = results["samples"]["variables"]["w"]


for i in range(N):
    fig, ax = plt.subplots(1,1)
    ax.hist(w_rms[:,i])
    # ax[1].hist(w[:,i])
    plt.title("This is for weight "+str(w[i]))
    plt.show()
    print("mean of weight="+str(w[i])+" is "+str(jnp.mean(w_rms[:,i]))+", w_true="+str(w[i]))

exit()
# w1 vs w2
w1d = onp.linspace(0.00, 0.20, 100)
w2d = onp.linspace(-0.70, -0.30, 100)
W1d, W2d = onp.meshgrid(w1d, w2d)
p12d = onp.vstack([W1d.ravel(), W2d.ravel()])

plt.figure()
plt.title("w_1 vs w_2 (rms)")

# plt.xlim([min(w_rms[:,0]), max(w_rms[:,0])])
# plt.ylim([min(w_rms[:,1]), max(w_rms[:,1])])
plt.plot(w_rms[:, 0], w_rms[:, 1], 'o', alpha=0.5, markersize=5, zorder=-1)
plt.plot(w[:, 0], w[:, 1], 'x', c='red',alpha=0.5, markersize=5, zorder=-1)
plt.show()
# w3 vs w4
w3d = onp.linspace(-0.3, -0.05, 100)
w4d = onp.linspace(-0.75, -0.575, 100)
W3d, W4d = onp.meshgrid(w3d, w4d)
p34d = onp.vstack([W3d.ravel(), W4d.ravel()])

plt.figure()
plt.title("w_3 vs w_4 (rms)")
plt.plot(w_rms[:, 2], w_rms[:, 3], 'o', alpha=0.5, markersize=5, zorder=-1)
plt.plot(w[:, 2], w[:, 3], 'x', c='red', alpha=0.5, markersize=5, zorder=-1)
plt.show()