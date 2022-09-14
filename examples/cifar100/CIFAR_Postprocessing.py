import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str('1')  # needs to stay before importing jax

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

(train_images, train_labels), _ = \
    tf.keras.datasets.cifar10.load_data()

import tensorflow_datasets as tfds
from jax_sgmc import data
from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader

train_dataset, train_info = tfds.load('Cifar10',
                                      split='train',
                                      with_info=True)
train_loader = TensorflowDataLoader(train_dataset,
                                    shuffle_cache=1000,
                                    exclude_keys=['id'])

train_batch_fn = data.random_reference_data(train_loader, cached_batches, batch_size)

batch_init, batch_get, batch_release = train_batch_fn
_, batch_data = batch_get(batch_init(), information=True)
init_batch, info_batch = batch_data

def init_resnet():
    @hk.transform_with_state
    def resnet(batch, is_training=True):
        images = batch["image"].astype(jnp.float32) / 255.
        resnet50 = hk.nets.ResNet50(num_classes)
        logits = resnet50(images, is_training=is_training)
        return logits
    return resnet.init, resnet.apply

init, _ = init_resnet()
init_params, _ = init(random.PRNGKey(0), init_batch)

sample = {"w": init_params}


filename = '/home/student/ana/jax-sgmc/examples/cifar100/results'
custom_loader = HDF5Loader(filename, '', sample)


