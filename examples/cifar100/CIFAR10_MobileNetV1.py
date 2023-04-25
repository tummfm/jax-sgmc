import sys
import time
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = str('2')  # needs to stay before importing jax
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
iterations = int(1e+4)
adam_iterations = 900
batch_size = 2048 * 2
burn_in_size = int(7e+3)
lr_first = 0.001
lr_last = 5e-8
temperature_param = 1
accepted_samples = 40

CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = jnp.array([0.2023, 0.1994, 0.2010])

# Load dataset
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()

train_mean = np.mean(np.true_divide(train_images, 255, dtype=np.float32), axis=(0, 1, 2))
train_std = np.std(np.true_divide(train_images, 255, dtype=np.float32), axis=(0, 1, 2))

test_mean = np.mean(np.true_divide(test_images, 255, dtype=np.float32), axis=(0, 1, 2))
test_std = np.std(np.true_divide(test_images, 255, dtype=np.float32), axis=(0, 1, 2))

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

# zeros_init_batch = train_loader.initializer_batch(batch_size)
batch_state, batch_data = train_batch_get(init_train_data_state, information=True)
init_batch, info_batch = batch_data
test_init_state, test_init_batch = test_batch_get(test_batch_init(), information=True)
val_init_state, val_init_batch = test_batch_get(val_batch_init(), information=True)


# MobileNet Model
def init_mobilenet():
    # @hk.without_apply_rng
    @hk.transform
    def mobilenetv1(batch, is_training=True):
        images = batch["image"].astype(jnp.float32)
        mobilenet = hk.nets.MobileNetV1(num_classes=num_classes, use_bn=False)
        logits = mobilenet(images, is_training=is_training)
        return logits

    return mobilenetv1.init, mobilenetv1.apply


init, apply_mobilenet = init_mobilenet()
# apply_mobilenet = jit(apply_mobilenet)
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


max_likelihood = False
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
    training_loader = NumpyDataLoader(image=train_images[:-10000, :, :, :], label=train_labels[:-10000, :])
    train_batch_init, train_batch_get, train_release = data.random_reference_data(training_loader, cached_batches,
                                                                                  batch_size)
    temp = train_batch_init(shuffle=False, in_epochs=False)
    first_batch_state, first_batch_data = train_batch_get(temp, information=True)
    first_batch, first_info_batch = first_batch_data
    logits = apply_mobilenet(params, None, init_batch)
    target_labels = onp.array(init_batch['label'])
    mini_batch_state = first_batch_state
    accuracy = []
    correct_count = 0
    incorrect_count = 0
    for i in range(train_images.shape[0] // batch_size):  # go over minibatch of training data
        mini_batch_state, mini_batch = train_batch_get(mini_batch_state, information=False)
        temp_labels = onp.array(mini_batch['label'])
        temp_logits = apply_mobilenet(params, None, mini_batch, is_training=False)
        argmax_results = onp.argmax(temp_logits, axis=-1)
        correct_count += sum(argmax_results == onp.squeeze(temp_labels))
        incorrect_count += sum(argmax_results != onp.squeeze(temp_labels))
    accuracy.append(np.true_divide(correct_count, (correct_count + incorrect_count)))
    print("Training accuracy=" + str(accuracy[0]))
    training_loader = NumpyDataLoader(image=test_images[5000:, :, :, :],
                                      label=test_labels[5000:, :])  # in practice it's a validation loader
    train_batch_init, train_batch_get, train_release = data.random_reference_data(training_loader, cached_batches,
                                                                                  batch_size)
    temp = train_batch_init(shuffle=True, in_epochs=True)
    first_batch_state, first_batch_data = train_batch_get(temp, information=True)
    first_batch, first_info_batch = first_batch_data
    logits = apply_mobilenet(params, None, init_batch)
    target_labels = onp.array(init_batch['label'])
    mini_batch_state = first_batch_state
    accuracy = []
    correct_count = 0
    incorrect_count = 0
    for i in range(5000 // batch_size):  # go over minibatch of training data
        mini_batch_state, mini_batch = train_batch_get(mini_batch_state, information=False)
        temp_labels = onp.array(mini_batch['label'])
        temp_logits = apply_mobilenet(params, None, mini_batch, is_training=False)
        argmax_results = onp.argmax(temp_logits, axis=-1)
        correct_count += sum(argmax_results == onp.squeeze(temp_labels))
        incorrect_count += sum(argmax_results != onp.squeeze(temp_labels))
    accuracy.append(np.true_divide(correct_count, (correct_count + incorrect_count)))
    print("Validation accuracy=" + str(accuracy[0]))
    sys.exit()

prior_scale = 10.


def gaussian_prior(sample):
    prior_params = sample["w"]
    gaussian = partial(jscipy.stats.norm.logpdf, loc=0., scale=prior_scale)
    priors = tree_map(gaussian, prior_params)
    return tree_math.Vector(priors).sum()


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

def evaluate_model(results, loader, model="SGLD", validation=True, plot=True, fig_name="model1e+5iters"):
    my_parameter_mapper, sth_to_remove = data.full_data_mapper(loader, 1, 1000)


    # train_batch_init, train_batch_get, train_release = data.random_reference_data(loader, cached_batches,
    #                                                                               batch_size)
    # if (validation):
    #     temp = train_batch_init(shuffle=True, in_epochs=True)
    # else:
    #     if (model=="SGLD"):
    #         temp = train_batch_init(shuffle=False, in_epochs=False)
    #     else:
    #         temp = train_batch_init(shuffle=True, in_epochs=False)
    #
    # first_batch_state, first_batch_data = train_batch_get(temp, information=True)
    # mini_batch_state = first_batch_state
    accuracy = []

    @jit
    def calculate_accuracy(batch, mask, carry):
        temp_labels = onp.array(batch['label'])
        temp_logits = apply_mobilenet(params, None, batch, is_training=False)
        argmax_results = onp.argmax(temp_logits, axis=-1)
        correct_count = sum(argmax_results == onp.squeeze(temp_labels))
        incorrect_count = sum(argmax_results != onp.squeeze(temp_labels))
        return [correct_count, incorrect_count], carry+1

    for j in range(accepted_samples):  # go over parameter samples
        params = tree_map(lambda x: onp.array(x[j]), results['w'])
        out, _ = my_parameter_mapper(calculate_accuracy, 0, masking=True)
        accuracy.append(sum(out[0]/(sum(out[0])+sum(out[1]))))

        # logits = onp.empty(shape=(0, 10))
        # target_labels = onp.empty(shape=(0, 1))
        # for i in range(train_images.shape[0] // batch_size):  # go over minibatch of training data
        #     mini_batch_state, mini_batch = train_batch_get(mini_batch_state, information=False)
        #     temp_labels = onp.array(mini_batch['label'])
        #     target_labels = onp.concatenate([target_labels, onp.array(mini_batch['label'])], axis=0)
        #     temp_logits = apply_mobilenet(params, None, mini_batch)
        #     logits = onp.concatenate([logits, temp_logits], axis=0)
        #     argmax_results = onp.argmax(temp_logits, axis=-1)
        #     correct_count += sum(argmax_results == onp.squeeze(temp_labels))
        #     incorrect_count += sum(argmax_results != onp.squeeze(temp_labels))
        # if (j == 0):
        #     logits_full = onp.expand_dims(logits, axis=0)
        #     target_labels_full = onp.expand_dims(target_labels, axis=0)
        # else:
        #     logits_full = onp.concatenate([onp.expand_dims(logits, axis=0), logits_full], axis=0)
        #     target_labels_full = onp.concatenate([onp.expand_dims(target_labels, axis=0), target_labels_full], axis=0)
        # accuracy.append(correct_count / (correct_count + incorrect_count))
    # pred_result, target = logits_full, target_labels_full
    if plot:
        plt.plot(onp.arange(1, len(accuracy) + 1, step=1), accuracy)
        plt.xlabel("num of sampled params")
        plt.ylabel("accuracy "+validation*"validation"+(not validation)*"training")
        plt.savefig(fig_name)
        plt.show()
    # return pred_result, target

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

    training_loader = NumpyDataLoader(image=train_images[:-10000, :, :, :], label=train_labels[:-10000, :])
    param_loader = NumpyDataLoader(results)
    evaluate_model(results, training_loader, validation=False)
    # evaluate_model(results, param_loader, validation=False)
    validation_loader = NumpyDataLoader(image=test_images[5000:, :, :, :],
                                      label=test_labels[5000:, :])
    evaluate_model(results, validation_loader, validation=True)
