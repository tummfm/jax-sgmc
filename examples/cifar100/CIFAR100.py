import pickle
from jax import nn, tree_leaves, random, numpy as jnp
from jax_sgmc import data, potential, adaption, scheduler, integrator, solver, io
import tensorflow as tf
import tensorflow_datasets
import haiku as hk
from jax_sgmc import alias


## Configuration parameters

batch_size = 16
cached_batches = 1024
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
                                         batch_size,
                                         shuffle_cache=1000,
                                         exclude_keys=['id'])

test_loader = data.NumpyDataLoader(X=test_images, Y=test_labels)

train_batch_fn = data.random_reference_data(train_loader, cached_batches)

# get first batch to init NN
# TODO: Maybe write convenience function for this common usecase?
batch_init, batch_get = train_batch_fn
# This method returns a batch with correct shape but all zero values. The batch
# contains 16 (batch_size) images.
init_batch = train_loader.initializer_batch(batch_size)

## ResNet Model


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

# print(jnp.sum(logits))
# I don't think this should give plain 0, otherwise gradients will be 0

## Initialize potential

def likelihood(resnet_state, sample, observations):
    labels = nn.one_hot(observations["label"], num_classes)
    logits, resnet_state = apply_resnet(sample["w"], resnet_state, None, observations)
    softmax_xent = labels * jnp.log(nn.softmax(logits))
    softmax_xent = -jnp.sum(softmax_xent, axis=-1)  # xent = (categorical) cross entropy
    softmax_xent /= labels.shape[0]
    return softmax_xent, resnet_state

def prior(sample):
    # Implement weight decay, corresponds to Gaussian prior over weights
    weights = sample["w"]
    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in tree_leaves(weights))
    return weight_decay * l2_loss

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
                                             is_batched=True)

## Setup Integrator

# Number of iterations: Ca. 0.035 seconds per iteration (including saving)
iterations = 100000

solver_sgld = alias.sgld(potential_fn=potential_fn, data_loader=train_loader, batch_size=batch_size)

sample = {"w": init_params}

results = solver_sgld(sample, iterations=iterations)[0]['samples']['variables']

# Simple pickle the results for now

with open("results.pkl", "wb") as file:
    pickle.dump(results, file)

print("Finished")