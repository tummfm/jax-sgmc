# Copyright 2021 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


from jax import jit, random, numpy as jnp, scipy as jscipy, tree_map
from jax_sgmc import data, potential, alias
from jax_sgmc.data.numpy_loader import NumpyDataLoader
import tensorflow as tf
import haiku as hk
import optax
import tree_math
from functools import partial
import numpy as onp
import matplotlib.pyplot as plt

onp.random.seed(123)
tf.random.set_seed(123)
key = random.PRNGKey(123)

tf.config.set_visible_devices([], device_type="GPU")

# Configuration parameters
cached_batches = 10
num_classes = 10

# Load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Rescaling images
train_images = tf.image.resize(
    train_images,
    (112, 112),
    method=tf.image.ResizeMethod.BILINEAR,
    preserve_aspect_ratio=True
)
test_images = tf.image.resize(
    test_images,
    (112, 112),
    method=tf.image.ResizeMethod.BILINEAR,
    preserve_aspect_ratio=True
)

# Hyper-parameters
batch_size = 256
epochs = 200
iterations_per_epoch = int(train_images.shape[0] / batch_size)
iterations = epochs * iterations_per_epoch
burn_in_size = (epochs - 20) * iterations_per_epoch
lr_first = 0.001
gamma = 0.33
lr_last = lr_first * (iterations) ** (-gamma)
accepted_samples = 20

# Split data and organize into DataLoaders
train_loader = NumpyDataLoader(image=train_images, label=onp.squeeze(train_labels))
train_batch_fn = data.random_reference_data(train_loader, cached_batches, batch_size)
test_loader = NumpyDataLoader(image=test_images[:test_labels.shape[0] // 2, ::],
                              label=test_labels[:test_labels.shape[0] // 2, :])
val_loader = NumpyDataLoader(image=test_images[test_labels.shape[0] // 2:, ::],
                             label=test_labels[test_labels.shape[0] // 2:, :])
test_batch_init, test_batch_get, test_release = data.random_reference_data(test_loader, cached_batches, batch_size)
val_batch_init, val_batch_get, val_release = data.random_reference_data(val_loader, cached_batches, batch_size)

# Get first batch to initialize NN
train_batch_init, train_batch_get, _ = train_batch_fn
init_train_data_state = train_batch_init()
batch_state, batch_data = train_batch_get(init_train_data_state, information=True)
init_batch, info_batch = batch_data
val_init_state, val_init_batch = test_batch_get(val_batch_init(), information=True)
test_init_state, test_init_batch = test_batch_get(test_batch_init(), information=True)


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

# Sanity-check prediction
logits = apply_mobilenet(init_params, None, init_batch)


# Initialize potential with log-likelihood
def log_likelihood(sample, observations):
    logits = apply_mobilenet(sample["w"], None, observations)
    # Log-likelihood is negative cross entropy
    log_likelihood = -optax.softmax_cross_entropy_with_integer_labels(logits, observations["label"])
    return log_likelihood


# Set gaussian prior
prior_scale = 10.
def log_gaussian_prior(sample):
    prior_params = sample["w"]
    gaussian = partial(jscipy.stats.norm.logpdf, loc=0., scale=prior_scale)
    priors = tree_map(gaussian, prior_params)
    return tree_math.Vector(priors).sum()

# Construct potential
potential_fn = potential.minibatch_potential(prior=log_gaussian_prior,
                                             likelihood=log_likelihood,
                                             is_batched=True,
                                             strategy='vmap')

# Define sample (of model parameters)
sample = {"w": init_params}

# Sanity-check likelihoods
_, returned_likelihoods = potential_fn(sample, batch_data, likelihoods=True)

# Create pSGLD sampler (with RMSProp precodntioner)
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

# Perform sampling
results = sampler(sample, iterations=iterations)
results = results[0]['samples']['variables']


# Define model evaluation
def evaluate_model(results, loader, evaluation="hard", dataset="training"):
    my_parameter_mapper, sth_to_remove = data.full_data_mapper(loader, 1, 1000)

    @jit
    def fetch_logits(batch, mask, carry):
        temp_logits = apply_mobilenet(params, None, batch, is_training=False)
        return temp_logits, carry + 1

    if dataset == "validation" or dataset == "test":
        logits_all = onp.empty((accepted_samples, test_labels.shape[0] // 2, num_classes))
    else:
        logits_all = onp.empty((accepted_samples, train_labels.shape[0], num_classes))
    # Go over sampled parameters (NNs)
    for j in range(accepted_samples):
        params = tree_map(lambda x: jnp.array(x[j]), results['w'])
        # Collect logits
        out, _ = my_parameter_mapper(fetch_logits, 0, masking=True)
        logits_all[j, :, :] = out.reshape(-1, 10)

    # Hard-voting: obtain predicted class labels from each model and use the most frequently predicted class
    if evaluation == "hard":
        class_predictions = onp.argmax(logits_all, axis=-1)
        hard_class_predictions = onp.apply_along_axis(lambda x: onp.bincount(x, minlength=10).argmax(), 0,
                                                      class_predictions)
        if dataset == "validation":
            accuracy = sum(hard_class_predictions == onp.squeeze(test_labels[test_labels.shape[0] // 2:, :])) / (
                        test_labels.shape[0] // 2)
        elif dataset == "test":
            accuracy = sum(hard_class_predictions == onp.squeeze(test_labels[:test_labels.shape[0] // 2, :])) / (
                        test_labels.shape[0] // 2)
        else:
            accuracy = sum(hard_class_predictions == onp.squeeze(train_labels)) / train_labels.shape[0]
        print((dataset == "training") * "Training" + (dataset == "validation") * "Validation" + (
                dataset == "test") * "Test" + " Hard-Voting Accuracy: " + str(
            accuracy * 100) + "%")

        # Calculating certainty (per image)
        certainty = onp.count_nonzero(class_predictions == hard_class_predictions, axis=0) / accepted_samples

        # Evaluating accuracy when certainty is above a fixed threshold
        accuracy_over_certainty = []
        for k in range(6):
            if dataset == "validation":
                accuracy_over_certainty.append(
                    sum(hard_class_predictions[onp.asarray(certainty >= (0.5 + 0.1 * k))] ==
                        onp.squeeze(onp.asarray(test_labels[test_labels.shape[0] // 2:, :]))[
                            onp.asarray(certainty >= (0.5 + 0.1 * k))]) / sum(certainty >= (0.5 + 0.1 * k)) * 100)
            elif dataset == "test":
                accuracy_over_certainty.append(
                    sum(hard_class_predictions[onp.asarray(certainty >= (0.5 + 0.1 * k))] ==
                        onp.squeeze(onp.asarray(test_labels[:test_labels.shape[0] // 2, :]))[
                            onp.asarray(certainty >= (0.5 + 0.1 * k))]) / sum(certainty >= (0.5 + 0.1 * k)) * 100)
            else:
                accuracy_over_certainty.append(
                    sum(hard_class_predictions[onp.asarray(certainty >= (0.5 + 0.1 * k))] == onp.squeeze(
                        train_labels[onp.asarray(certainty >= (0.5 + 0.1 * k))])) / sum(
                        certainty >= (0.5 + 0.1 * k)) * 100)
        print((dataset == "training") * "Training" + (dataset == "validation") * "Validation" + (
                dataset == "test") * "Test" + " Hard-Voting Accuracy-Over-Certainty: " + str(
            accuracy_over_certainty) + "%")

    # Soft-voting: obtain predicted probabilities from each model use the mean of these probabilities to pick a class
    probabilities = onp.exp(logits_all)
    mean_probabilities = onp.mean(probabilities, axis=0)
    soft_class_predictions = onp.argmax(mean_probabilities, axis=-1)
    if dataset == "validation":
        accuracy = sum(soft_class_predictions == onp.squeeze(test_labels[test_labels.shape[0] // 2:, :])) / (
                    test_labels.shape[0] // 2)
        random_samples = onp.random.randint(0, test_labels.shape[0] // 2 - 1, 5)
    elif dataset == "test":
        accuracy = sum(soft_class_predictions == onp.squeeze(test_labels[:test_labels.shape[0] // 2, :])) / (
                    test_labels.shape[0] // 2)
        random_samples = onp.random.randint(0, test_labels.shape[0] // 2 - 1, 5)
    else:
        accuracy = sum(soft_class_predictions == onp.squeeze(train_labels)) / train_labels.shape[0]
        random_samples = onp.random.randint(0, train_labels.shape[0] - 1, 5)
    print((dataset == "training") * "Training" + (dataset == "validation") * "Validation" + (
            dataset == "test") * "Test" + " Soft-Voting Accuracy: " + str(accuracy * 100) + "%")

    # Plotting five randomly chosen samples
    fig, ax = plt.subplots(1, 5, figsize=(10.8, 5))
    for i in range(len(random_samples)):
        ax[i].boxplot(probabilities[:, random_samples[i], :])
        ax[i].set_title("Image " + str(random_samples[i]))
        ax[i].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    fig.tight_layout(pad=0.2)
    plt.savefig("UQ_CIFAR10_" + (dataset == "training") * "Training" + (dataset == "validation") * "Validation" + (
            dataset == "test") * "Test" + ".pdf", format="pdf")
    plt.show()


# Model evaluation
evaluate_model(results, train_loader, dataset="training")
evaluate_model(results, val_loader, dataset="validation")
evaluate_model(results, test_loader, dataset="test")
