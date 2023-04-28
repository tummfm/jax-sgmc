import sys
import time
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = str('0')  # needs to stay before importing jax
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from jax import jit, random, numpy as jnp, scipy as jscipy, value_and_grad, tree_map, numpy as onp
from jax_sgmc import data, potential, adaption, scheduler, integrator, solver, io, alias
import tensorflow as tf
import tensorflow_datasets as tfds
import haiku as hk
import optax
import tree_math
import h5py
import matplotlib.pyplot as plt
from jax_sgmc.data.numpy_loader import NumpyDataLoader
import jax
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
tf.random.set_seed(123)
key = random.PRNGKey(123)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Configuration parameters
cached_batches = 1
num_classes = 10
weight_decay = 5.e-4

# parameters
iterations = int(5e+5)
adam_iterations = 200
batch_size = 2048 * 2
burn_in_size = int(4e+5)
lr_first = 0.001
lr_last = 5e-8
temperature_param = 1
accepted_samples = 40

# Load dataset
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()

# Use tensorflow dataset directly. The 'id' must be excluded as text is not
# supported by jax
dataset, train_info = tfds.load('Cifar10',
                                split=['train[:70%]', 'test[70%:]'],
                                with_info=True)
train_dataset, test_dataset = dataset
train_loader = NumpyDataLoader(
    image=train_images,
    label=np.squeeze(train_labels))
train_batch_fn = data.random_reference_data(train_loader, cached_batches,
                                            batch_size)

# ana: test data not needed here, but keeping the code nonetheless
test_loader = NumpyDataLoader(image=test_images[:5000, :, :, :], label=test_labels[:5000, :])
val_loader = NumpyDataLoader(image=test_images[5000:, :, :, :], label=test_labels[5000:, :])
test_batch_init, test_batch_get, test_release = data.random_reference_data(test_loader, cached_batches, batch_size)
val_batch_init, val_batch_get, val_release = data.random_reference_data(val_loader, cached_batches, batch_size)

# get first batch to init NN
# TODO: Maybe write convenience function for this common usecase?
train_batch_init, train_batch_get, _ = train_batch_fn

init_train_data_state = train_batch_init()

batch_state, batch_data = train_batch_get(init_train_data_state, information=True)
init_batch, info_batch = batch_data
test_init_state, test_init_batch = test_batch_get(test_batch_init(), information=True)
val_init_state, val_init_batch = test_batch_get(val_batch_init(), information=True)


# MobileNet Model
def init_mobilenet():
    @hk.transform
    def mobilenetv1(batch, is_training=True):
        images = batch["image"].astype(jnp.float32)
        mobilenet = hk.nets.MobileNetV1(num_classes=num_classes, use_bn=False)
        logits = mobilenet(images, is_training=is_training)
        return logits

    return mobilenetv1.init, mobilenetv1.apply


init, apply_mobilenet = init_mobilenet()
apply_mobilenet = jit(apply_mobilenet)
init_params = init(key, init_batch)

# sanity-check prediction
logits = apply_mobilenet(init_params, None, init_batch)


# Initialize potential with log-likelihood
def likelihood(sample, observations):
    logits = apply_mobilenet(sample["w"], None, observations)
    # log-likelihood is negative cross entropy
    log_likelihood = -optax.softmax_cross_entropy_with_integer_labels(
        logits, observations["label"])
    return log_likelihood


max_likelihood = True
if max_likelihood:
    # loss is negative log_likelihood
    def loss_fn(params, batch):
        logits = apply_mobilenet(params, None, batch)
        cross_entropies = optax.softmax_cross_entropy_with_integer_labels(
            logits, jnp.squeeze(batch["label"]))
        return jnp.mean(cross_entropies)


    lr_schedule = optax.exponential_decay(-lr_first, adam_iterations, 0.1)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule)
    )

    params = init_params
    train_data_state = init_train_data_state
    opt_state = optimizer.init(init_params)
    loss_list = []
    for i in range(adam_iterations):
        train_data_state, batch = train_batch_get(train_data_state,
                                                  information=False)

        loss, grad = value_and_grad(loss_fn)(params, batch)
        scaled_grad, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, scaled_grad)

        if i % 1 == 0:
            loss_list.append(loss)
            print(f'Loss at iteration {i}: {loss}')
    plt.plot(loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('mobilenet_adam_1.png')
    plt.show()


    def evaluate_model(params, loader, validation=True):
        my_parameter_mapper, sth_to_remove = data.full_data_mapper(loader, 1, 1000)

        @jit
        def calculate_accuracy(batch, mask, carry):
            temp_labels = onp.array(batch['label'])
            temp_logits = apply_mobilenet(params, None, batch, is_training=False)
            argmax_results = onp.argmax(temp_logits, axis=-1)
            correct_count = sum(argmax_results == onp.squeeze(temp_labels))
            incorrect_count = sum(argmax_results != onp.squeeze(temp_labels))
            return [correct_count, incorrect_count], carry + 1

        out, _ = my_parameter_mapper(calculate_accuracy, 0, masking=True)
        accuracy = (sum(out[0] / (sum(out[0]) + sum(out[1]))))
        print("Validation"*validation+"Training"*(not validation)+" accuracy = "+str(accuracy))

    evaluate_model(params, train_loader, validation=False)
    evaluate_model(params, val_loader)

prior_scale = 10.

def gaussian_prior(sample):
    prior_params = sample["w"]
    gaussian = partial(jscipy.stats.norm.logpdf, loc=0., scale=prior_scale)
    priors = tree_map(gaussian, prior_params)
    return tree_math.Vector(priors).sum()


# imporper prior
def prior(sample):
    return jnp.array(1.0, dtype=jnp.float32)


# The likelihood accepts a batch of data, so no batching strategy is required, instead, is_batched must be set to true.
# The likelihood signature changes from:   (Sample, Data) -> Likelihood
#                                   to :   (State, Sample, Data) -> Likelihood, NewState
# if has_state is set to true.
potential_fn = potential.minibatch_potential(prior=gaussian_prior,
                                             likelihood=likelihood,
                                             is_batched=True,
                                             strategy='vmap',
                                             temperature=temperature_param)

sample = {"w": init_params, "std": jnp.array([1.0])}

test_prior = gaussian_prior(sample)

_, returned_likelihoods = potential_fn(sample, batch_data, likelihoods=True)

use_alias = True
if use_alias:
    sampler = alias.sgld(potential_fn=potential_fn,
                         data_loader=train_loader,
                         cache_size=cached_batches,
                         batch_size=batch_size,
                         first_step_size=lr_first,
                         last_step_size=lr_last,
                         burn_in=burn_in_size,
                         accepted_samples=accepted_samples,
                         rms_prop=True,
                         progress_bar=True)
    results_original = sampler(sample, iterations=iterations)
    results = results_original[0]
    results = results['samples']['variables']
    params = tree_map(lambda x: onp.array(x[0]), results['w'])


    def evaluate_model(results, loader, validation=True, plot=True, fig_name="mod1e+5iters"):
        my_parameter_mapper, sth_to_remove = data.full_data_mapper(loader, 1, 1000)
        accuracy = []

        @jit
        def calculate_accuracy(batch, mask, carry):
            temp_labels = onp.array(batch['label'])
            temp_logits = apply_mobilenet(params, None, batch, is_training=False)
            argmax_results = onp.argmax(temp_logits, axis=-1)
            correct_count = sum(argmax_results == onp.squeeze(temp_labels))
            incorrect_count = sum(argmax_results != onp.squeeze(temp_labels))
            return [correct_count, incorrect_count], carry + 1

        for j in range(accepted_samples):  # go over parameter samples
            params = tree_map(lambda x: onp.array(x[j]), results['w'])
            out, _ = my_parameter_mapper(calculate_accuracy, 0, masking=True)
            accuracy.append(sum(out[0] / (sum(out[0]) + sum(out[1]))))

        if plot:
            plt.plot(onp.arange(1, len(accuracy) + 1, step=1), accuracy)
            plt.xlabel("Number of sampled parameters")
            plt.ylabel(validation * "Validation" + (not validation) * "Training" + " accuracy")
            plt.savefig(fig_name+validation * "Validation" + (not validation) * "Training" )
            plt.show()


    training_loader = NumpyDataLoader(image=train_images[:, :, :, :], label=train_labels[:, :])
    evaluate_model(results, training_loader, validation=False, fig_name="mobilenet_5e+5_lr5e-3")
    validation_loader = NumpyDataLoader(image=test_images[5000:, :, :, :],
                                        label=test_labels[5000:, :])
    evaluate_model(results, validation_loader, validation=True, fig_name="mobilenet_5e+5_lr5e-3")
