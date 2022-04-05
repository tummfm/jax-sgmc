import pickle
import time

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str('1')  # needs to stay before importing jax

from jax import nn, tree_leaves, random, numpy as jnp
from jax_sgmc import data, potential, adaption, scheduler, integrator, solver, io, alias
import tensorflow as tf
import tensorflow_datasets
import haiku as hk
import numpy as onp
from jax.scipy import stats as stats
from jax import lax, tree_map, tree_leaves, numpy as jnp
from jax import scipy as jscipy
# import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution
#
# disable_eager_execution()

# from jax import device_put, devices
# print(device_put(1, devices()[3]).device_buffer.device())
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# config = tf.config.experimental.set_memory_growth(physical_devices[1], True)
# config = tf.config.experimental.set_memory_growth(physical_devices[2], True)
# config = tf.config.experimental.set_memory_growth(physical_devices[3], True)

# if len(sys.argv) > 1:
#     visible_device = str(sys.argv[1])
# else:
#     visible_device = 3
# os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)  # controls on which gpu the program runs

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

## Configuration parameters

batch_size = 64
cached_batches = 512
num_classes = 100
weight_decay = 5.e-4

## Load dataset

(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar100.load_data(label_mode='fine')

# Use tensorflow dataset directly. The 'id' must be excluded as text is not
# supported by jax
train_dataset, train_info = tensorflow_datasets.load('Cifar100',
                                                     split='train',
                                                     with_info=True)
train_loader = data.TensorflowDataLoader(train_dataset,
                                         shuffle_cache=1000,
                                         exclude_keys=['id'])

test_loader = data.NumpyDataLoader(image=test_images, label=test_labels)

train_batch_fn = data.random_reference_data(train_loader, cached_batches, batch_size)

test_batch_init, test_batch_get = data.random_reference_data(test_loader, cached_batches, batch_size)

# get first batch to init NN
# TODO: Maybe write convenience function for this common usecase?
batch_init, batch_get = train_batch_fn

# This method returns a batch with correct shape but all zero values. The batch
# contains 16 (batch_size) images.
zeros_init_batch = train_loader.initializer_batch(batch_size)  # ana: it doesn't help if I change this to ones
_, batch_data = batch_get(batch_init(), information=True)
init_batch, info_batch = batch_data
_, test_init_batch = test_batch_get(test_batch_init(), information=True)


# ResNet Model
def init_resnet():
    @hk.transform_with_state
    def resnet(batch, is_training=True):
        images = batch["image"].astype(jnp.float32) / 255.
        resnet50 = hk.nets.ResNet50(num_classes)
        logits = resnet50(images, is_training=is_training)
        return logits
    return resnet.init, resnet.apply

init, apply_resnet = init_resnet()
init_params, init_resnet_state = init(random.PRNGKey(0), init_batch)

# test prediction
logits, _ = apply_resnet(init_params, init_resnet_state, None, init_batch)

print(jnp.sum(logits))
# I don't think this should give plain 0, otherwise gradients will be 0


# Initialize potential with log-likelihood
def likelihood(model_state, sample, observations):
    labels = nn.one_hot(observations["label"], num_classes)
    logits, new_state = apply_resnet(sample["w"], model_state, None, observations)
    softmax_xent = labels * jnp.log(nn.softmax(logits))
    softmax_xent = jnp.mean(softmax_xent, axis=1)
    softmax_xent /= labels.shape[0]
    likelihood = jnp.zeros(64, dtype=jnp.float32)
    if 'image' in observations.keys():  # if-condition probably not even necessary here
        likelihood += jscipy.stats.norm.logpdf(observations['label']-softmax_xent, scale=sample['std'])
    return likelihood, new_state


# def prior(sample):
#     # Implement weight decay, corresponds to Gaussian prior over weights
#     weights = sample["w"]
#     l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in tree_leaves(weights))
#     return weight_decay * l2_loss

def prior(sample):
    # return jscipy.stats.expon.pdf(sample['w'])
    return jnp.array(1.0, dtype=jnp.float32)

# The likelihood accepts a batch of data, so no batching strategy is required,
# instead, is_batched must be set to true.
#
# The likelihood signature changes from
#   (Sample, Data) -> Likelihood
# to
#   (State, Sample, Data) -> Likelihood, NewState
# if has_state is set to true.
potential_fn = potential.minibatch_potential(prior=prior,
                                             likelihood=likelihood,
                                             has_state=True,  # likelihood contains model state
                                             is_batched=True,
                                             strategy='vmap')  # or change to pmap

sample = {"w": init_params, "std": jnp.array([1.0])}

_, (returned_likelihoods, _) = potential_fn(sample, batch_data, likelihoods=True)

# Setup Integrator
# Number of iterations: Ca. 0.035 seconds per iteration (including saving)
# iterations = 100000
iterations = 4000

rms_prop = adaption.rms_prop()
rms_integrator = integrator.langevin_diffusion(potential_fn,
                                               train_batch_fn,
                                               rms_prop)

# Schedulers
rms_step_size = scheduler.polynomial_step_size_first_last(first=1e-6,
                                                          last=5e-7)  # try smaller ones
# burn_in = scheduler.initial_burn_in(5000)
burn_in = scheduler.initial_burn_in(500) # try burn-in = 0
# Has ca. 23.000.000 parameters, so not more than 500 samples fit into RAM
rms_random_thinning = scheduler.random_thinning(rms_step_size, burn_in, 500)

rms_scheduler = scheduler.init_scheduler(step_size=rms_step_size,
                                         burn_in=burn_in,
                                         thinning=rms_random_thinning)

# tf = "results"
import h5py
from jax_sgmc import io
import h5py
file = h5py.File('results', "w")
# f = h5py.File("mytestfile.hdf5", "w")
# data_collector = io.MemoryCollector(file)
data_collector = io.HDF5Collector(file)
saving = io.save(data_collector)
# data_collector = io.MemoryCollector()
# saving = io.save(data_collector)

# dc = io.MemoryCollector(save_dir="results")
# saving = io.save(dc)
rms_sgld = solver.sgmc(rms_integrator)
rms_run = solver.mcmc(rms_sgld,
                      rms_scheduler,
                      saving=saving)

rms_integ = rms_integrator[0](sample,
                              init_model_state=init_resnet_state)
start = time.time()
rms_results = rms_run(rms_integ,
                      iterations=iterations)  #["samples"]["variables"]
file.create_dataset('samples', data=rms_results[0]['samples'])
print(f"Run completed in {time.time() - start} seconds")
if file:
    file.close()
# file.close()

print("Finished")
