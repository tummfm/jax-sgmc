import contextlib
import logging
import time
import timeit

import functools
import os

from typing import Iterable, Mapping, NamedTuple, Tuple

import tree

os.environ["CUDA_VISIBLE_DEVICES"] = str('0')  # needs to stay before importing jax

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
batch_size = 128
cached_batches = 1
num_classes = 10
weight_decay = 5.e-4
deterministic_weight_decay = 1e-4
train_smoothing = 0.1
eval_batch_size = 1000
train_epochs = 90

## Load dataset
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()
# tf.keras.datasets.cifar100.load_data(label_mode='fine')

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

# sanity-check prediction
logits, state = apply_resnet(init_params, init_resnet_state, None, init_batch)


# Initialize potential with log-likelihood
def likelihood(model_state, sample, observations):
    labels = nn.one_hot(observations["label"], num_classes)
    logits, model_state = apply_resnet(sample["w"], model_state, None, observations)
    softmax_xent = jnp.multiply(labels, jnp.log(nn.softmax(logits)))
    likelihood = jnp.sum(softmax_xent)
    # TODO ask Stephan: Is the rest of this necessary?
    # softmax_xent = jnp.sum(softmax_xent, axis=1)
    # softmax_xent /= labels.shape[0]
    # likelihood = jnp.zeros(batch_size, dtype=jnp.float32)
    # if 'image' in observations.keys():  # if-condition probably not even necessary here
        # check the distribution that corresponds to a softmax -> it does
        # likelihood += jscipy.stats.norm.logpdf(observations['label'] - softmax_xent, scale=sample['std'])
        # likelihood -= softmax_xent
        # likelihood += jscipy.stats.norm.logpdf(labels - softmax_xent, scale=sample['std'])
        # TODO check if the scale is a random-walk (if yes - sample there, if no - more burn-in)

    return likelihood, model_state


# def prior(sample):
#     # Implement weight decay, corresponds to Gaussian prior over weights
#     weights = sample["w"]
#     # l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in tree_leaves(weights))
#     prior = 0
#     for p in tree_flatten(weights):
#         i=0
#         # TODO exclude batch norm (see line 141 in train.py in resnet example in Haiku)
#     prior += jscipy.stats.norm.logpdf([p for ((mod_name, _), p) in tree_flatten(weights)
#          if 'batchnorm' not in mod_name], 0, weight_decay)
#     return prior


def prior(sample):
    # return jscipy.stats.expon.pdf(sample['w'])
    return jnp.array(1.0, dtype=jnp.float32)


# The likelihood accepts a batch of data, so no batching strategy is required, instead, is_batched must be set to true.
# The likelihood signature changes from:   (Sample, Data) -> Likelihood
#                                   to :   (State, Sample, Data) -> Likelihood, NewState
# if has_state is set to true.
potential_fn = potential.minibatch_potential(prior=prior,
                                             likelihood=likelihood,
                                             has_state=True,  # likelihood contains model state
                                             is_batched=True,
                                             strategy='vmap')  # or change to pmap
sample = {"w": init_params, "std": jnp.array([1.0])}
_, (returned_likelihoods, returned_state) = potential_fn(sample, batch_data, likelihoods=True)

# vanilla training with an SGD optimizer
Scalars = Mapping[str, jnp.ndarray]


# class TrainState(NamedTuple):
#     params: hk.Params
#     state: hk.State
#     opt_state: optax.OptState
#     loss_scale: jmp.LossScale


get_policy = lambda: jmp.get_policy('p=f32,c=f32,o=f32')
get_bn_policy = lambda: jmp.get_policy('p=f32,c=f32,o=f32')


# def get_initial_loss_scale() -> jmp.LossScale:
#   cls = getattr(jmp, 'NoOpLossScale')
#   return cls(2 ** 15) if cls is not jmp.NoOpLossScale else cls()

# def _forward(
#     batch,
#     is_training: bool,
# ) -> jnp.ndarray:
#   """Forward application of the resnet."""
#   images = batch['images']
#   net = hk.nets.ResNet50(1000,
#                          resnet_v2=True,
#                          bn_config={'decay_rate': 0.9})
#   return net(images, is_training=is_training)
#
# # Transform our forwards function into a pair of pure functions.
# forward = hk.transform_with_state(_forward)


# def lr_schedule(step: jnp.ndarray) -> jnp.ndarray:
#   """Cosine learning rate schedule."""
#   # train_split = dataset.Split.from_string('TRAIN_AND_VALID')
#
#   total_batch_size = 128 * jax.device_count()
#   steps_per_epoch = train_labels.shape[0] / total_batch_size
#   warmup_steps = 5 * steps_per_epoch
#   training_steps = 90 * steps_per_epoch
#
#   lr = 0.1 * total_batch_size / 256
#   scaled_step = (jnp.maximum(step - warmup_steps, 0) /
#                  (training_steps - warmup_steps))
#   lr *= 0.5 * (1.0 + jnp.cos(jnp.pi * scaled_step))
#   if warmup_steps:
#     lr *= jnp.minimum(step / warmup_steps, 1.0)
#   return lr

def make_optimizer() -> optax.GradientTransformation:
    """SGD with nesterov momentum and a custom lr schedule."""
    return optax.chain(
        optax.trace(
            decay=0.9,
            nesterov=True),
        optax.scale(-1))

def l2_loss(params: Iterable[jnp.ndarray]) -> jnp.ndarray:
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


def loss_fn(
        params: hk.Params,
        state: hk.State,
        loss_scale: jmp.LossScale,
        batch,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State]]:
    """Computes a regularized loss for the given batch."""
    logits, state = apply_resnet(params, state, None, batch, is_training=True)
    labels = nn.one_hot(batch['labels'], num_classes)
    if train_smoothing:
        labels = optax.smooth_labels(labels, train_smoothing)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
    l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params)
    # l2_params = [p for ((mod_name, _), p) in tree_util.tree_flatten(params)
                 if 'batchnorm' not in mod_name]
    loss = loss + deterministic_weight_decay * l2_loss(l2_params)
    return loss_scale.scale(loss), (loss, state)


# @functools.partial(pmap, axis_name='i', donate_argnums=(0,))
def train_step(
        train_state,
        # batch,
) -> _:
    """Applies an update to parameters and returns new state."""

    # TODO: ask Stephan
    #   How can I apply resnet and update params (get samples) for one step?
    #   Where should burn-in be done?

    potential_fn = potential.minibatch_potential(prior=prior,
                                                 likelihood=likelihood,
                                                 has_state=True,  # likelihood contains model state
                                                 is_batched=True,
                                                 strategy='vmap')  # or change to pmap

    # grads_jax_sgmc, (returned_likelihoods, returned_state) = potential_fn(sample, batch_data, likelihoods=True)

    # params, state, opt_state, loss_scale = train_state
    # grads, (loss, new_state) = (
    #     grad(loss_fn, has_aux=True)(params, state, loss_scale, batch))
    next_train_state, batch_data = batch_get(train_state)
    grads, (loss, new_state) = potential_fn(sample, batch_data, likelihoods=True)

    # Grads are in "param_dtype" (likely F32) here. We cast them back to the
    # compute dtype such that we do the all-reduce below in the compute precision
    # (which is typically lower than the param precision).
    policy = get_policy()
    # grads = policy.cast_to_compute(grads)
    # grads = loss_scale.unscale(grads)

    # Taking the mean across all replicas to keep params in sync.
    grads = lax.pmean(grads, axis_name='i')

    # We compute our optimizer update in the same precision as params, even when
    # doing mixed precision training.
    grads = policy.cast_to_param(grads)

    # Compute and apply updates via our optimizer.
    # updates, new_opt_state = make_optimizer().update(grads, opt_state)
    # new_params = optax.apply_updates(params, updates)

    # if FLAGS.mp_skip_nonfinite:
    #     grads_finite = jmp.all_finite(grads)
    #     loss_scale = loss_scale.adjust(grads_finite)
    #     new_params, new_state, new_opt_state = jmp.select_tree(
    #         grads_finite,
    #         (new_params, new_state, new_opt_state),
    #         (params, state, opt_state))

    # Scalars to log (note: we log the mean across all hosts/devices).
    # scalars = {'train_loss': loss, 'loss_scale': loss_scale.loss_scale}
    # if FLAGS.mp_skip_nonfinite:
    #   scalars['grads_finite'] = grads_finite
    # new_state, scalars = jmp.cast_to_full((new_state, scalars))
    # scalars = lax.pmean(scalars, axis_name='i')
    # train_state = TrainState(new_params, new_state, new_opt_state, loss_scale)
    return new_state


# def initial_state(rng: jnp.ndarray, batch) -> TrainState:
#   """Computes the initial network state."""
#   params, state = forward.init(rng, batch, is_training=True)
#   opt_state = make_optimizer().init(params)
#   loss_scale = get_initial_loss_scale()
#   return TrainState(params, state, opt_state, loss_scale)

@jax.jit
def eval_batch(
        params: hk.Params,
        state: hk.State,
        batch,
) -> jnp.ndarray:
    """Evaluates a batch."""
    logits, _ = apply_resnet(params, state, None, batch, is_training=False)
    predicted_label = jnp.argmax(logits, axis=-1) # TODO: take care of how predictions are obtained
    correct = jnp.sum(jnp.equal(predicted_label, batch['labels']))
    return correct.astype(jnp.float32)

test_dataset = test_init_batch
def evaluate(
    split,
    params: hk.Params,
    state: hk.State,
    test_dataset_state=test_init_state,
) -> Scalars:
  """Evaluates the model at the given params/state."""
  if split.num_examples % eval_batch_size:
    raise ValueError(f'Eval batch size {eval_batch_size} must be a '
                     f'multiple of {split} num examples {split.num_examples}')

  # Params/state are sharded per-device during training. We just need the copy
  # from the first device (since we do not pmap evaluation at the moment).
  params, state = tree_util.tree_map(lambda x: x[0], (params, state))
  # test_dataset = dataset.load(split,
  #                             is_training=False,
  #                             batch_dims=[eval_batch_size],
  #                             transpose=FLAGS.dataset_transpose,
  #                             zeros=FLAGS.dataset_zeros)
  test_dataset_state, test_dataset = test_batch_get(test_dataset_state)
  correct = jnp.array(0)
  total = 0
  for batch in test_dataset:
    correct += eval_batch(params, state, batch)
    total += batch['labels'].shape[0]
  assert total == split.num_examples, total
  return {'top_1_acc': correct.item() / total}

@contextlib.contextmanager
def time_activity(activity_name: str):
  logging.info('[Timing] %s start.', activity_name)
  start = timeit.default_timer()
  yield
  duration = timeit.default_timer() - start
  logging.info('[Timing] %s finished (Took %.2fs).', activity_name, duration)

train_split = train_labels.shape[0]
eval_split = 10000 #TODO replace with actual variable
total_train_batch_size = 128 * jax.device_count()
num_train_steps = ((len(train_dataset) * train_epochs) // batch_size)
mp_policy = get_policy()
bn_policy = get_bn_policy().with_output_dtype(mp_policy.compute_dtype)
hk.mixed_precision.set_policy(hk.BatchNorm, bn_policy)
hk.mixed_precision.set_policy(hk.nets.ResNet50, mp_policy)

# batch = next(train_dataset)
train_state = batch_state
# train_state = TrainState()
for step_num in range(num_train_steps):
    # batch_state, batch_data = batch_get(batch_state)
    train_state= train_step(train_state)



# Setup Integrator
# Number of iterations: Ca. 0.035 seconds per iteration (including saving)
iterations = 100000

rms_prop = adaption.rms_prop()
rms_integrator = integrator.langevin_diffusion(potential_fn,
                                               train_batch_fn,
                                               rms_prop)

# Schedulers
rms_step_size = scheduler.polynomial_step_size_first_last(first=5e-7,
                                                          # a good starting point is 1e-3, start sampling at 1e-6
                                                          last=5e-8)

burn_in = scheduler.initial_burn_in(
    10000)  # large burn-in: if you need 100k for deterministic training, then 200k burn-in

# Has ca. 23.000.000 parameters, so not more than 500 samples fit into RAM
rms_random_thinning = scheduler.random_thinning(rms_step_size, burn_in, 250)

rms_scheduler = scheduler.init_scheduler(step_size=rms_step_size,
                                         burn_in=burn_in,
                                         thinning=rms_random_thinning)

with h5py.File('results_iterations_100k_burn_in_10k_lr_5e7_prior_weight_decay_bs_128', "w") as file:
    data_collector = io.HDF5Collector(file)
    saving = io.save(data_collector)

    rms_sgld = solver.sgmc(rms_integrator)
    rms_run = solver.mcmc(rms_sgld,
                          rms_scheduler,
                          saving=saving)
    rms_integ = rms_integrator[0](sample,
                                  init_model_state=init_resnet_state)

    start = time.time()
    rms_results = rms_run(rms_integ,
                          iterations=iterations)

print("Finished")
