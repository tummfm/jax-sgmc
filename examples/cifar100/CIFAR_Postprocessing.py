import sys

import h5py
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str('1')  # needs to stay before importing jax

from jax import nn, tree_leaves, random
import tensorflow as tf
import tensorflow_datasets
import haiku as hk
import numpy as onp
from jax import lax, tree_map, tree_leaves, tree_flatten, numpy as jnp

from jax_sgmc.data.hdf5_loader import HDF5Loader

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = jnp.array([0.2023, 0.1994, 0.2010])

batch_size = 32
cached_batches = 1
num_classes = 10
weight_decay = 5.e-4

(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()

import numpy as np
train_images = np.true_divide(train_images, 255, dtype=np.float32)
train_mean = np.mean(train_images, axis=(0,1,2))
train_std = np.std(train_images, axis=(0,1,2))

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
training_loader = DeviceNumpyDataLoader(image=train_images[:-10000, :, :, :], label=train_labels[:-10000, :])



def init_mobilenet():
    @hk.transform
    def mobilenetv1(batch, is_training=True):
        images = jnp.true_divide((batch["image"].astype(jnp.float32) - train_mean), train_std)
        mobilenet = hk.nets.MobileNetV1(num_classes=num_classes, use_bn=False)
        logits = mobilenet(images, is_training=True)
        return logits

    return mobilenetv1.init, mobilenetv1.apply


validation_batch_fn = data.random_reference_data(validation_loader, cached_batches, batch_size)
train_batch_fn = data.random_reference_data(training_loader, cached_batches, batch_size)

# batch_init, batch_get, batch_release = validation_batch_fn
batch_init, batch_get, batch_release = train_batch_fn
# zeros_init_batch = validation_loader.initializer_batch(batch_size)
zeros_init_batch = training_loader.initializer_batch(batch_size)
_, batch_data = batch_get(batch_init(), information=True)
init_batch, info_batch = batch_data
init, apply_resnet = init_mobilenet()
init_params = init(random.PRNGKey(0), init_batch)
sample = {"w": init_params}

# from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader

# validation_loader = TensorflowDataLoader(validation_dataset,
#                                          shuffle_cache=1000,
#                                          exclude_keys=['id'])


# val_batch_init, val_batch_get, val_release = data.random_reference_data(validation_loader, cached_batches, batch_size)
train_batch_init, train_batch_get, train_release = data.random_reference_data(training_loader, cached_batches, batch_size)

# temp = val_batch_init(shuffle=False, in_epochs=True)
# first_batch_state, first_batch_data = val_batch_get(temp, information=True)
temp = train_batch_init(shuffle=False, in_epochs=True)
first_batch_state, first_batch_data = train_batch_get(temp, information=True)
first_batch, first_info_batch = first_batch_data
target_labels = np.empty([0, 0])


def map_fn(batch, mask, carry):
    params = batch["w"]
    params = tree_map(lambda x: jnp.array(x[0]), params)
    logits = apply_resnet(params, None, first_batch)
    target_labels = jnp.array(first_batch['label'])
    mini_batch_state = first_batch_state
    # for i in range(validation_loader.static_information['observation_count'] // batch_size):
    for i in range(training_loader.static_information['observation_count'] // batch_size):
    # for i in range(6):
        mini_batch_state, mini_batch = train_batch_get(mini_batch_state)
        target_labels = jnp.concatenate([target_labels, jnp.array(mini_batch['label'])], axis=0)
        logits = jnp.concatenate([logits, apply_resnet(params, None, mini_batch)], axis=0)
    return [logits, target_labels], carry + 1

filepath = '/home/student/ana/jax-sgmc/examples/cifar100/mobilenet_1'
with h5py.File(filepath, "r") as file:
    postprocess_loader = HDF5Loader(
        filepath,
        subdir="/chain~0/variables",
        sample=sample)

    my_parameter_mapper, _ = data.full_data_mapper(postprocess_loader, 1, 1)
    out, _ = my_parameter_mapper(map_fn, 0, masking=True)
    result, target = out
    target_labels_array = onp.array(jnp.mean(target, axis=0))
    import matplotlib.pyplot as plt

    results_np_array = onp.array(result[:, 1, :])

    accuracy = []

    correct_count = 0
    incorrect_count = 0
    for i in range(result.shape[0]):
        sliced_results = (onp.array(result[:i+1, :, :]))
        argmax_results = onp.argmax(sliced_results, axis=-1)
        for j in range(argmax_results.shape[1]):
            correct_count += argmax_results[i][j]==int(target_labels_array[j])
            incorrect_count += argmax_results[i][j]!=int(target_labels_array[j])
        accuracy.append(correct_count / (correct_count + incorrect_count))

    plt.plot(accuracy)
    plt.xlabel("num of sampled params")
    plt.ylabel("accuracy")
    plt.savefig("accuracy_plot_mobilenet_1.png")
    plt.show()

exit()
from jax_sgmc.data.numpy_loader import DeviceNumpyDataLoader

test_loader = DeviceNumpyDataLoader(image=test_images, label=test_labels)

test_batch_init, test_batch_get, test_release = data.random_reference_data(test_loader, cached_batches, batch_size)

# _, test_init_batch = test_batch_get(test_batch_init(), information=True)


# actually getting the first batch
temp = test_batch_init(shuffle=True, in_epochs=True)
first_batch_state, first_batch_data = test_batch_get(temp, information=True)
first_batch, first_info_batch = first_batch_data
