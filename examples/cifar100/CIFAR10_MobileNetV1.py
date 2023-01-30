import contextlib
import logging
import time
import timeit

import functools
import os

from typing import Iterable, Mapping, NamedTuple, Tuple

import tree

os.environ["CUDA_VISIBLE_DEVICES"] = str('1')  # needs to stay before importing jax

from jax import jit, nn, pmap, grad, tree_util, tree_leaves, tree_flatten, tree_map, lax, random, numpy as jnp, \
    scipy as jscipy
from jax_sgmc import data, potential, adaption, scheduler, integrator, solver, io, alias
import tensorflow as tf
import tensorflow_datasets as tfds
import haiku as hk
import numpy as onp
import optax
import jax
import jmp
import tree_math
import h5py
from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader
from jax_sgmc.data.numpy_loader import DeviceNumpyDataLoader

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# if len(sys.argv) > 1:
#     visible_device = str(sys.argv[1])
# else:
#     visible_device = 3
# os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)  # controls on which gpu the program runs

## Configuration parameters

cached_batches = 1
num_classes = 10
weight_decay = 5.e-4

# parameters
iterations = int(1e+6)
batch_size = 128
burn_in_size = int(1e+5)
lr_first = 5e-7
lr_last = 5e-8

CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = jnp.array([0.2023, 0.1994, 0.2010])

## Load dataset
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()

# Use tensorflow dataset directly. The 'id' must be excluded as text is not
# supported by jax
train_dataset, train_info = tfds.load('Cifar10',
                                      split=['train[:70%]', 'test[70%:]'],
                                      with_info=True)
train_dataset, validation_dataset = train_dataset
train_loader = TensorflowDataLoader(train_dataset,
                                    shuffle_cache=1000,
                                    exclude_keys=['id'])

# ana: test data not needed here, but keeping the code nonetheless
test_loader = DeviceNumpyDataLoader(image=test_images, label=test_labels)
train_batch_fn = data.random_reference_data(train_loader, cached_batches, batch_size)
test_batch_init, test_batch_get, test_release = data.core.random_reference_data(test_loader, cached_batches, batch_size)

# get first batch to init NN
# TODO: Maybe write convenience function for this common usecase?
batch_init, batch_get, batch_release = train_batch_fn

zeros_init_batch = train_loader.initializer_batch(batch_size)  # ana: it doesn't help if I change this to ones
batch_state, batch_data = batch_get(batch_init(), information=True)
init_batch, info_batch = batch_data
test_init_state, test_init_batch = test_batch_get(test_batch_init(), information=True)


# MobileNet Model
def init_mobilenet():
    @hk.transform
    def mobilenetv1(batch, is_training=True):
        # images = batch["image"].astype(jnp.float32) / 255.
        images = (batch["image"].astype(jnp.float32) - CIFAR10_MEAN) / CIFAR10_STD
        mobilenet = hk.nets.MobileNetV1(num_classes=num_classes, use_bn=False)
        logits = mobilenet(images, is_training=True)
        return logits

    return mobilenetv1.init, mobilenetv1.apply


init, apply_mobilenet = init_mobilenet()
init_params = init(random.PRNGKey(0), init_batch)

# sanity-check prediction
logits = apply_mobilenet(init_params, None, init_batch)


# Initialize potential with log-likelihood
def likelihood(sample, observations):
    logits = apply_mobilenet(sample["w"], None, observations)
    # log-likelihood is negative cross entropy
    log_likelihood = -optax.softmax_cross_entropy_with_integer_labels(
        logits, observations["label"])
    return log_likelihood

prior_scale = 1.
def gaussian_prior(sample):
    params = tree_math.Vector(sample["w"])
    log_pdf = tree_math.unwrap(jscipy.stats.norm.logpdf, vector_argnums=0)(
        params, loc=0., scale=prior_scale)
    return log_pdf.sum()


def prior(sample):
    return jnp.array(1.0, dtype=jnp.float32)


# The likelihood accepts a batch of data, so no batching strategy is required, instead, is_batched must be set to true.
# The likelihood signature changes from:   (Sample, Data) -> Likelihood
#                                   to :   (State, Sample, Data) -> Likelihood, NewState
# if has_state is set to true.
potential_fn = potential.minibatch_potential(prior=prior,
                                             likelihood=likelihood,
                                             is_batched=True,
                                             strategy='vmap')  # or change to pmap
sample = {"w": init_params, "std": jnp.array([1.0])}
_, returned_likelihoods = potential_fn(sample, batch_data, likelihoods=True)





rms_prop = adaption.rms_prop()
rms_integrator = integrator.langevin_diffusion(potential_fn,
                                               train_batch_fn,
                                               rms_prop)

# Schedulers
rms_step_size = scheduler.polynomial_step_size_first_last(first=lr_first,
                                                          # a good starting point is 1e-3, start sampling at 1e-6
                                                          last=lr_last)

burn_in = scheduler.initial_burn_in(
    burn_in_size)  # large burn-in: if you need 100k for deterministic training, then 200k burn-in

# Has ca. 23.000.000 parameters, so not more than 500 samples fit into RAM
rms_random_thinning = scheduler.random_thinning(rms_step_size, burn_in, 50)

rms_scheduler = scheduler.init_scheduler(step_size=rms_step_size,
                                         burn_in=burn_in,
                                         thinning=rms_random_thinning)

with h5py.File('mobilenet_5', "w") as file:
    data_collector = io.HDF5Collector(file)
    saving = io.save(data_collector)

    rms_sgld = solver.sgmc(rms_integrator)
    rms_run = solver.mcmc(rms_sgld,
                          rms_scheduler,
                          saving=saving)
    rms_integ = rms_integrator[0](sample)

    start = time.time()
    rms_results = rms_run(rms_integ,
                          iterations=iterations)

print("Finished")
