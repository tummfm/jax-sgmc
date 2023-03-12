import itertools
import sys
import time
import os

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
from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader
from jax_sgmc.data.numpy_loader import NumpyDataLoader
import jax

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# if len(sys.argv) > 1:
#     visible_device = str(sys.argv[1])
# else:
#     visible_device = 3
# os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)  # controls on which gpu the program runs

# Configuration parameters

cached_batches = 1
num_classes = 10
weight_decay = 5.e-4

# parameters
iterations = int(1e+5)
batch_size = 1024
burn_in_size = int(5e+3)
lr_first = 0.001
lr_last = 5e-8
temperature_param = 0.01
accepted_samples = 40

CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = jnp.array([0.2023, 0.1994, 0.2010])

# Load dataset
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()

import numpy as np

# train_images = np.true_divide(train_images, 255, dtype=np.float32)
train_mean = np.mean(np.true_divide(train_images, 255, dtype=np.float32), axis=(0, 1, 2))
train_std = np.std(np.true_divide(train_images, 255, dtype=np.float32), axis=(0, 1, 2))

# Use tensorflow dataset directly. The 'id' must be excluded as text is not
# supported by jax
dataset, train_info = tfds.load('Cifar10',
                                split=['train[:70%]', 'test[70%:]'],
                                with_info=True)
train_dataset, test_dataset = dataset
train_loader = NumpyDataLoader(image=train_images, label=np.squeeze(train_labels))
# train_loader = TensorflowDataLoader(train_dataset,
#                                     shuffle_cache=1000,
#                                     exclude_keys=['id'])
train_batch_fn = data.random_reference_data(train_loader, cached_batches,
                                            batch_size)

# ana: test data not needed here, but keeping the code nonetheless
test_loader = NumpyDataLoader(image=test_images, label=test_labels)
test_batch_init, test_batch_get, test_release = data.random_reference_data(test_loader, cached_batches, batch_size)

# get first batch to init NN
# TODO: Maybe write convenience function for this common usecase?
train_batch_init, train_batch_get, _ = train_batch_fn

init_train_data_state = train_batch_init()

# zeros_init_batch = train_loader.initializer_batch(batch_size)
batch_state, batch_data = train_batch_get(init_train_data_state, information=True)
init_batch, info_batch = batch_data
test_init_state, test_init_batch = test_batch_get(test_batch_init(), information=True)


# MobileNet Model
def init_mobilenet():
    # @hk.without_apply_rng
    @hk.transform
    def mobilenetv1(batch, is_training=True):
        # images = batch["image"].astype(jnp.float32) / 255.
        images =  jnp.true_divide((batch["image"].astype(jnp.float32) - train_mean), train_std)
        # images =  batch["image"].astype(jnp.float32)
        mobilenet = hk.nets.MobileNetV1(num_classes=num_classes, use_bn=False)
        logits = mobilenet(images, is_training=True)
        return logits

    return mobilenetv1.init, mobilenetv1.apply


init, apply_mobilenet = init_mobilenet()
# apply_mobilenet = jit(apply_mobilenet)
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


max_likelihood = False
if max_likelihood:
    # loss is negative log_likelihood
    def loss_fn(params, batch):
        logits = apply_mobilenet(params, None, batch)
        cross_entropies = optax.softmax_cross_entropy_with_integer_labels(
            logits, jnp.squeeze(batch["label"]))
        return jnp.mean(cross_entropies)


    lr_schedule = optax.exponential_decay(-0.001, iterations, 0.1)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule)
    )

    params = init_params
    train_data_state = init_train_data_state
    opt_state = optimizer.init(init_params)
    loss_list = []
    for i in range(2000):
        train_data_state, batch = train_batch_get(train_data_state,
                                                  information=False)

        loss, grad = value_and_grad(loss_fn)(params, batch)
        scaled_grad, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, scaled_grad)

        if i % 1 == 0:
            loss_list.append(loss)
            print(f'Loss at iteration {i}: {loss}')
    plt.plot(loss_list)
    plt.savefig('mobilenet_adam_1.png')
    plt.show()
    from jax_sgmc.data.numpy_loader import DeviceNumpyDataLoader

    training_loader = DeviceNumpyDataLoader(image=train_images[:-10000, :, :, :], label=train_labels[:-10000, :])
    train_batch_fn = data.random_reference_data(training_loader, cached_batches, batch_size)
    batch_init, batch_get, batch_release = train_batch_fn
    # zeros_init_batch = validation_loader.initializer_batch(batch_size)
    zeros_init_batch = training_loader.initializer_batch(batch_size)
    _, batch_data = batch_get(batch_init(), information=True)
    init_batch, info_batch = batch_data
    init, apply_resnet = init_mobilenet()
    init_params = init(random.PRNGKey(0), init_batch)
    sample = {"w": init_params}
    train_batch_init, train_batch_get, train_release = data.random_reference_data(training_loader, cached_batches,
                                                                                  batch_size)
    temp = train_batch_init(shuffle=False, in_epochs=True)
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
        # target_labels = onp.concatenate([target_labels, onp.array(mini_batch['label'])], axis=0)
        temp_logits = apply_mobilenet(params, None, mini_batch)
        # logits = onp.concatenate([logits, temp_logits], axis=0)
        argmax_results = onp.argmax(temp_logits, axis=-1)
        correct_count += sum(argmax_results == onp.squeeze(temp_labels))
        incorrect_count += sum(argmax_results != onp.squeeze(temp_labels))
    accuracy.append(np.true_divide(correct_count, (correct_count + incorrect_count)))
    print(accuracy)
    sys.exit()

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
                                             strategy='vmap',
                                             temperature=temperature_param)

sample = {"w": init_params, "std": jnp.array([1.0])}
# batch_data[0]['label'] = jnp.squeeze(batch_data[0]['label'])
_, returned_likelihoods = potential_fn(sample, batch_data, likelihoods=True)

use_alias = True
if use_alias:
    # sampler = alias.sgld(potential_fn, train_loader, cache_size=cached_batches, batch_size=batch_size,
    #                      burn_in=burn_in_size, accepted_samples=accepted_samples, rms_prop=True, progress_bar=True)
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
    from jax_sgmc.data.numpy_loader import DeviceNumpyDataLoader

    training_loader = DeviceNumpyDataLoader(image=train_images[:-10000, :, :, :], label=train_labels[:-10000, :])
    train_batch_fn = data.random_reference_data(training_loader, cached_batches, batch_size)
    batch_init, batch_get, batch_release = train_batch_fn
    # zeros_init_batch = validation_loader.initializer_batch(batch_size)
    zeros_init_batch = training_loader.initializer_batch(batch_size)
    _, batch_data = batch_get(batch_init(), information=True)
    init_batch, info_batch = batch_data
    init, apply_resnet = init_mobilenet()
    init_params = init(random.PRNGKey(0), init_batch)
    sample = {"w": init_params}
    train_batch_init, train_batch_get, train_release = data.random_reference_data(training_loader, cached_batches,
                                                                                  batch_size)
    temp = train_batch_init(shuffle=False, in_epochs=True)
    first_batch_state, first_batch_data = train_batch_get(temp, information=True)
    first_batch, first_info_batch = first_batch_data
    logits = apply_mobilenet(params, None, init_batch)
    target_labels = onp.array(init_batch['label'])
    mini_batch_state = first_batch_state

    pytree_structure = jax.tree_structure(sample)
    reference_data = [key_tuple for key_tuple in io.pytree_dict_keys(sample)]
    sample_format = tree_map(
        lambda leaf: jax.ShapeDtypeStruct(shape=leaf.shape, dtype=leaf.dtype),
        sample)
    observations_counts = [len(results[leaf_name[0]][leaf_name[1]][leaf_name[2]])
                           for leaf_name in reference_data]
    observation_count = observations_counts[0]
    selected_observations = []

    accuracy = []
    correct_count = 0
    incorrect_count = 0
    for j in range(accepted_samples):  # go over parameter samples
        params = tree_map(lambda x: onp.array(x[j]), results['w'])
        logits = onp.empty(shape=(0, 10))
        target_labels = onp.empty(shape=(0, 1))
        for i in range(train_images.shape[0] // batch_size):  # go over minibatch of training data
            mini_batch_state, mini_batch = train_batch_get(mini_batch_state, information=False)
            temp_labels = onp.array(mini_batch['label'])
            target_labels = onp.concatenate([target_labels, onp.array(mini_batch['label'])], axis=0)
            temp_logits = apply_mobilenet(params, None, mini_batch)
            logits = onp.concatenate([logits, temp_logits], axis=0)
            argmax_results = onp.argmax(temp_logits, axis=-1)
            correct_count += sum(argmax_results == onp.squeeze(temp_labels))
            incorrect_count += sum(argmax_results != onp.squeeze(temp_labels))
        if (j == 0):
            logits_full = onp.expand_dims(logits, axis=0)
            target_labels_full = onp.expand_dims(target_labels, axis=0)
        else:
            logits_full = onp.concatenate([onp.expand_dims(logits, axis=0), logits_full], axis=0)
            target_labels_full = onp.concatenate([onp.expand_dims(target_labels, axis=0), target_labels_full], axis=0)
        accuracy.append(correct_count / (correct_count + incorrect_count))
    pred_result, target = logits_full, target_labels_full
    # target_labels_array = onp.array(onp.mean(target, axis=0))
    import matplotlib.pyplot as plt

    plt.plot(onp.arange(1, len(accuracy) + 1, step=1), accuracy)
    plt.xlabel("num of sampled params")
    plt.ylabel("accuracy")
    plt.savefig("accuracy_plot_mobilenet_100000iters.png")
    plt.show()
    exit()

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

with h5py.File('mobilenet_2', "w") as file:
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
