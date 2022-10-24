import sys

import h5py
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str('0')  # needs to stay before importing jax

from jax import nn, tree_leaves, random, numpy as jnp
import tensorflow as tf
import tensorflow_datasets
import haiku as hk
import numpy as onp
from jax import lax, tree_map, tree_leaves, numpy as jnp

from jax_sgmc.data.hdf5_loader import HDF5Loader

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 64
cached_batches = 1
num_classes = 10
weight_decay = 5.e-4

(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()

import tensorflow_datasets as tfds
from jax_sgmc import data
from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader

train_dataset, train_info = tfds.load('Cifar10',
                                      split=['train[:70%]', 'test[70%:]'],
                                      with_info=True)  # shuffling is disabled by default
train_dataset, validation_dataset = train_dataset

validation_images, validation_labels = train_images[-10000:, :, :, :], train_labels[-10000:, :]
from jax_sgmc.data.numpy_loader import DeviceNumpyDataLoader

validation_loader = DeviceNumpyDataLoader(image=validation_images, label=validation_labels)


def init_resnet():
    @hk.transform_with_state
    def resnet(batch, is_training=True):
        images = batch["image"].astype(jnp.float32) / 255.
        resnet50 = hk.nets.ResNet50(num_classes)
        logits = resnet50(images, is_training=is_training)
        return logits

    return resnet.init, resnet.apply


validation_batch_fn = data.random_reference_data(validation_loader, cached_batches, batch_size)

batch_init, batch_get, batch_release = validation_batch_fn
zeros_init_batch = validation_loader.initializer_batch(batch_size)
_, batch_data = batch_get(batch_init(), information=True)
init_batch, info_batch = batch_data
init, apply_resnet = init_resnet()
init_params, init_resnet_state = init(random.PRNGKey(0), init_batch)
sample = {"w": init_params}

# from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader

# validation_loader = TensorflowDataLoader(validation_dataset,
#                                          shuffle_cache=1000,
#                                          exclude_keys=['id'])


val_batch_init, val_batch_get, val_release = data.random_reference_data(validation_loader, cached_batches, batch_size)

temp = val_batch_init(shuffle=False, in_epochs=True)
first_batch_state, first_batch_data = val_batch_get(temp, information=True)
first_batch, first_info_batch = first_batch_data
target_labels = np.empty([0, 0])


def map_fn(batch, mask, carry):
    params = batch["w"]
    params = tree_map(lambda x: jnp.array(x[0]), params)
    logits, _ = apply_resnet(params, init_resnet_state, None, first_batch)
    target_labels = jnp.array(first_batch['label'])
    mini_batch_state = first_batch_state
    for i in range(validation_loader.static_information['observation_count'] // batch_size):
    # for i in range(4):
        mini_batch_state, mini_batch = val_batch_get(mini_batch_state)
        target_labels = jnp.concatenate([target_labels, jnp.array(mini_batch['label'])], axis=0)
        logits = jnp.concatenate([logits, apply_resnet(params, init_resnet_state, None, mini_batch)[0]], axis=0)
    return [logits, target_labels], carry + 1


with h5py.File('/home/student/ana/jax-sgmc/examples/cifar100/results_iterations_100k_burn_in_10k_lr_5e7_prior_weight_decay', "r") as file:
    postprocess_loader = HDF5Loader(
        '/home/student/ana/jax-sgmc/examples/cifar100/results_iterations_100k_burn_in_10k_lr_5e7_prior_weight_decay',
        subdir="/chain~0/variables",
        sample=sample)

    my_parameter_mapper, _ = data.full_data_mapper(postprocess_loader, 1, 1)
    out, _ = my_parameter_mapper(map_fn, 0, masking=True)
    result, target = out
    target_labels_array = onp.array(jnp.mean(target, axis=0))
    import scipy
    import matplotlib.pyplot as plt

    results_np_array = onp.array(result[:, 1, :])
    correct_count = 0
    incorrect_count = 0
    for i in range(250):
        sliced_results = (onp.array(result[:, i, :]))
        argmax_results = onp.argmax(sliced_results, axis=-1)
        argmax_probabilities = scipy.special.softmax(sliced_results)
        mean = onp.mean(result[:, i, :], axis=0)
        # count, np_bins = np.histogram(argmax_results)
        correct_count += onp.sum(argmax_results==target_labels_array[i])
        incorrect_count += onp.sum(argmax_results!=target_labels_array[i])
        # plt.hist(argmax_results, bins=np.arange(0, 10, 1))
        # plt.show()

    print("correct predictions: ", correct_count)
    print("incorrect predictions: ", incorrect_count)
    print("accuracy: ", correct_count/(correct_count+incorrect_count))
    i = 0
    # my_full_data_mapper, release = data.full_data_mapper(postprocess_loader, cached_batches_count=1, mb_size=1)

    # logits, _ = my_full_data_mapper(apply_resnet, (init_resnet_state, None, init_batch))

exit()
from jax_sgmc.data.numpy_loader import DeviceNumpyDataLoader

test_loader = DeviceNumpyDataLoader(image=test_images, label=test_labels)

test_batch_init, test_batch_get, test_release = data.random_reference_data(test_loader, cached_batches, batch_size)

# _, test_init_batch = test_batch_get(test_batch_init(), information=True)


# actually getting the first batch
temp = test_batch_init(shuffle=True, in_epochs=True)
first_batch_state, first_batch_data = test_batch_get(temp, information=True)
first_batch, first_info_batch = first_batch_data
