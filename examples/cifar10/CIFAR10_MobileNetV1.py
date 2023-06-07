import sys
import time
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = str('2')  # needs to stay before importing jax
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from jax import jit, random, numpy as jnp, scipy as jscipy, value_and_grad, tree_map
from jax_sgmc import data, potential, alias
import tensorflow as tf
import haiku as hk
import optax
import tree_math
from jax_sgmc.data.numpy_loader import NumpyDataLoader
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
cached_batches = 10
num_classes = 10

# Load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
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

# parameters
batch_size = 256
epochs = 15
# epochs = 1
iterations_per_epoch = int(train_images.shape[0]
                           / batch_size)
iterations = epochs * iterations_per_epoch
burn_in_size = iterations - 100
lr_first = 0.001
gamma = 0.55
lr_last = lr_first * (iterations) ** (-gamma)
accepted_samples = 20

train_loader = NumpyDataLoader(image=train_images, label=np.squeeze(train_labels))
train_batch_fn = data.random_reference_data(train_loader, cached_batches, batch_size)

# ana: test data not needed here, but keeping the code nonetheless
test_loader = NumpyDataLoader(image=test_images[:5000, :, :, :], label=test_labels[:5000, :])
val_loader = NumpyDataLoader(image=test_images[5000:, :, :, :], label=test_labels[5000:, :])
test_batch_init, test_batch_get, test_release = data.random_reference_data(test_loader, cached_batches, batch_size)
val_batch_init, val_batch_get, val_release = data.random_reference_data(val_loader, cached_batches, batch_size)

# get first batch to init NN
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

# sanity-check prediction
logits = apply_mobilenet(init_params, None, init_batch)


# Initialize potential with log-likelihood
def likelihood(sample, observations):
    logits = apply_mobilenet(sample["w"], None, observations)
    # log-likelihood is negative cross entropy
    log_likelihood = -optax.softmax_cross_entropy_with_integer_labels(logits, observations["label"])
    return log_likelihood


prior_scale = 10.

def gaussian_prior(sample):
    prior_params = sample["w"]
    gaussian = partial(jscipy.stats.norm.logpdf, loc=0., scale=prior_scale)
    priors = tree_map(gaussian, prior_params)
    return tree_math.Vector(priors).sum()


# The likelihood accepts a batch of data, so no batching strategy is required, instead, is_batched must be set to true.
# The likelihood signature changes from:   (Sample, Data) -> Likelihood
#                                   to :   (State, Sample, Data) -> Likelihood, NewState
# if has_state is set to true.
potential_fn = potential.minibatch_potential(prior=gaussian_prior,
                                             likelihood=likelihood,
                                             is_batched=True,
                                             strategy='vmap')

sample = {"w": init_params}

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


    def evaluate_model(results, loader, evaluation="hard", dataset="training"):
        my_parameter_mapper, sth_to_remove = data.full_data_mapper(loader, 1, 1000)

        # @jit
        def fetch_logits(batch, mask, carry):
            temp_logits = apply_mobilenet(params, None, batch, is_training=False)
            return temp_logits, carry + 1

        if dataset == "validation" or dataset == "test":
            logits_all = np.empty((accepted_samples, test_labels.shape[0]//2, num_classes))
        else:
            logits_all = np.empty((accepted_samples, train_labels.shape[0], num_classes))

        for j in range(accepted_samples):  # go over parameter samples
            params = tree_map(lambda x: jnp.array(x[j]), results['w'])
            out, _ = my_parameter_mapper(fetch_logits, 0, masking=True)
            logits_all[j, :, :] = out.reshape(-1, 10)

        # evaluation using hard-voting: obtain predicted class labels from each model and use the most frequently predicted class
        if evaluation == "hard":
            class_predictions = np.argmax(logits_all, axis=-1)
            hard_class_predictions = np.apply_along_axis(lambda x: np.bincount(x, minlength=10).argmax(), 0,
                                                         class_predictions)

            if dataset == "validation":
                accuracy = sum(hard_class_predictions == np.squeeze(test_labels[test_labels.shape[0]//2:, :])) / test_labels.shape[0]//2
            elif dataset == "test":
                accuracy = sum(hard_class_predictions == np.squeeze(test_labels[:test_labels.shape[0]//2, :])) / test_labels.shape[0]//2
            else:
                accuracy = sum(hard_class_predictions == np.squeeze(train_labels)) / train_labels.shape[0]
            print((dataset == "training") * "Training" + (dataset == "validation") * "Validation" + (
                        dataset == "test") * "Test" + " Hard-Voting Accuracy: " + str(
                accuracy * 100) + "%")

            certainty = np.count_nonzero(class_predictions == hard_class_predictions, axis=0) / accepted_samples
            accuracy_over_certainty = []
            for k in range(6):
                if dataset == "validation":
                    accuracy_over_certainty.append(
                        sum((certainty >= (0.5 + 0.1 * k)) * hard_class_predictions == np.squeeze(
                            test_labels[test_labels.shape[0]//2:, :])) / sum(certainty >= (0.5 + 0.1 * k)) * 100)
                elif dataset == "test":
                    accuracy_over_certainty.append(
                        sum((certainty >= (0.5 + 0.1 * k)) * hard_class_predictions == np.squeeze(
                            test_labels[:test_labels.shape[0]//2, :])) / sum(certainty >= (0.5 + 0.1 * k)) * 100)
                else:
                    accuracy_over_certainty.append(
                        sum((certainty >= (0.5 + 0.1 * k)) * hard_class_predictions == np.squeeze(train_labels)) / sum(
                            certainty >= (0.5 + 0.1 * k)) * 100)

            print((dataset == "training") * "Training" + (dataset == "validation") * "Validation" + (
                        dataset == "test") * "Test" + " Hard-Voting Accuracy-Over-Certainty: " + str(
                accuracy_over_certainty) + "%")


        # evaluation using soft-voting: obtain predicted probabilities from each model use the mean of these probabilities to pick a class
        probabilities = np.exp(logits_all)
        mean_probabilities = np.mean(probabilities, axis=0)
        soft_class_predictions = np.argmax(mean_probabilities, axis=-1)
        if dataset == "validation":
            accuracy = sum(soft_class_predictions == np.squeeze(test_labels[test_labels.shape[0]//2:, :])) / test_labels.shape[0]//2
            random_samples = np.random.randint(0, test_labels.shape[0]//2-1, 5)
        elif dataset == "test":
            accuracy = sum(soft_class_predictions == np.squeeze(test_labels[:test_labels.shape[0]//2, :])) / test_labels.shape[0]//2
            random_samples = np.random.randint(0, test_labels.shape[0]//2-1, 5)
        else:
            accuracy = sum(soft_class_predictions == np.squeeze(train_labels)) / train_labels.shape[0]
            random_samples = np.random.randint(0, train_labels.shape[0]-1, 5)
        print((dataset == "training") * "Training" + (dataset == "validation") * "Validation" + (
                    dataset == "test") * "Test" + " Soft-Voting Accuracy: " + str(accuracy * 100) + "%")

        fig, ax = plt.subplots(1, 5, figsize=(13, 4))
        for i in range(len(random_samples)):
            ax[i].boxplot(probabilities[:, random_samples[i], :])
            ax[i].set_title("Sample " + str(random_samples[i]))
            ax[i].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        fig.tight_layout(pad=0.2)
        plt.savefig("UQ_CIFAR10_" + (dataset == "training") * "Training" + (dataset == "validation") * "Validation" + (
                    dataset == "test") * "Test" + ".pdf", format="pdf")
        plt.show()

    evaluate_model(results, train_loader, dataset="training")
    evaluate_model(results, val_loader, dataset="validation")
    evaluate_model(results, test_loader, dataset="test")
