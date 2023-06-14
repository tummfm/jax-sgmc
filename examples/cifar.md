---
jupytext:
  formats: examples///ipynb,examples///md:myst,docs//examples//ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{raw-cell}

---
Copyright 2021 Multiscale Modeling of Fluid Materials, TU Munich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
---
```
# Image Classification on CIFAR-10

In this example we will show how _JaxSGMC_ can be used to set up and train a 
neural network. The objective is to perform image classification on the dataset 
[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) which consists of 60000 
32x32 images. We will use the [MobileNet](https://arxiv.org/abs/1704.04861) 
architecture implemented by [Haiku](https://github.com/deepmind/dm-haiku).

```python tags=["hide-cell"]
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
```

We set a seed for each library where we will use stochastic functionalities.

```python
onp.random.seed(123)
tf.random.set_seed(123)
key = random.PRNGKey(123)
```

Due to conflicts between JAX and TensorFlow we make sure that TensorFlow cannot 
see any GPU devices.

```python
tf.config.set_visible_devices([], device_type="GPU")
```

Now we continue by loading the data, setting hyper-parameters and rescaling the 
images from 32x32 to 112x112. The MobileNet architecture shows better 
performance with larger images. Then we also split the data and organize it 
into DataLoaders.

We try to balance between a large and a small (mini-)batch size since a larger 
choice usually leads to more robust updates while a smaller one leads to faster 
computation, in our view a (mini-)batch size of 256 is suitable. 
We wish to go over the full dataset 200 times, thus we calculate how many 
iterations are necessary depending on the chosen (mini-)batch size (here 39000 
iterations). We set the burn-in phase to cover 90% of the iterations and only 
consider samples from the final 10% of the iterations (here 35100 burn-in 
iterations). Here also thinning will be applied so that a fixed number of 
parameters is accepted - in our case 20 parameters are accepted.
For the learning rate (step size) we start with 0.001 (common choice for deep 
learning models) and calculate a final learning rate with a decay of 0.33.

```python
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
```

Now we split the data. 50000, 5000 and 5000 images are used as training, 
validation and test datasets.

```python
# Split data and organize into DataLoaders
train_loader = NumpyDataLoader(image=train_images, label=onp.squeeze(train_labels))
test_loader = NumpyDataLoader(image=test_images[:test_labels.shape[0] // 2, ::],
                              label=test_labels[:test_labels.shape[0] // 2, :])
val_loader = NumpyDataLoader(image=test_images[test_labels.shape[0] // 2:, ::],
                             label=test_labels[test_labels.shape[0] // 2:, :])
```

Now we need to obtain an initial batch of data such that the neural network can 
be initialized with a batch. The `random_reference_data` function initialized 
data access and allows randomly drawing mini-batches; it returns functions for 
initialization of a new reference data state, for getting a minibatch from the 
data state and for releasing the DataLoader once all computations have been done.

```python pycharm={"name": "#%%\n"}
# Initialize the random access to the training data
train_batch_init, train_batch_get, _ = data.random_reference_data(
  train_loader, cached_batches, batch_size)
  
init_train_data_state = train_batch_init()
batch_state, batch_data = train_batch_get(init_train_data_state, information=True)
init_batch, info_batch = batch_data

# Do the same for the valdation and test data
val_batch_init, val_batch_get, val_release = data.random_reference_data(
  val_loader, cached_batches, batch_size)
test_batch_init, test_batch_get, test_release = data.random_reference_data(
  test_loader, cached_batches, batch_size)
  
val_init_state, val_init_batch = test_batch_get(
  val_batch_init(), information=True)
test_init_state, test_init_batch = test_batch_get(
  test_batch_init(), information=True)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now the MobileNet architecture can be defined using the Haiku syntax.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
def init_mobilenet():
  @hk.transform
  def mobilenetv1(batch, is_training=True):
    images = batch["image"].astype(jnp.float32)
    mobilenet = hk.nets.MobileNetV1(num_classes=num_classes, use_bn=False)
    logits = mobilenet(images, is_training=is_training)
    return logits
  return mobilenetv1.init, mobilenetv1.apply
```

```python pycharm={"name": "#%%\n"}
init, apply_mobilenet = init_mobilenet()
apply_mobilenet = jit(apply_mobilenet)
init_params = init(key, init_batch)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
At this we test whether we can apply the Mobilenet network to a minibatch of 
data and if the obtained logits make sense.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Sanity-check prediction
logits = apply_mobilenet(init_params, None, init_batch)
print(logits)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now we define the log-likelihood and log-prior.
For multiclass classification the log-likelihood is the negative cross entropy.
We set a log gaussian prior centered at 0 and with a standard deviation of 10 
on the weights.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Initialize potential with log-likelihood
def log_likelihood(sample, observations):
  logits = apply_mobilenet(sample["w"], None, observations)
  
  # Log-likelihood is negative cross entropy
  log_likelihood = -optax.softmax_cross_entropy_with_integer_labels(
    logits, observations["label"])
    
  return log_likelihood


# Set gaussian prior
prior_scale = 10.
def log_gaussian_prior(sample):
  prior_params = sample["w"]
  gaussian = partial(jscipy.stats.norm.logpdf, loc=0., scale=prior_scale)
  priors = tree_map(gaussian, prior_params)
  return tree_math.Vector(priors).sum()
```

<!-- #region pycharm={"name": "#%% md\n"} -->
We have defined the log-likelihood to accept a batch of data, and we take care 
to set the `is_batched=True` when calling `minibatch_potential`.

We want to sample the neural network parameters; we denote them as `'w'` and use
the initial parameters as a starting sample.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
potential_fn = potential.minibatch_potential(prior=log_gaussian_prior,
                                             likelihood=log_likelihood,
                                             is_batched=True,
                                             strategy='vmap')

# Define sample (of model parameters)
sample = {"w": init_params}

# Sanity-check likelihoods
_, returned_likelihoods = potential_fn(sample, batch_data, likelihoods=True)
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now we use the `alias.py` module to set up a 
[pSGLD sampler with an RMSProp preconditioner](https://arxiv.org/abs/1512.07666).
The potential function, DataLoader for training, and a set of hyperparameters
need to be passed in order to initialize the sampler.
In this case a polynomial step size scheduler is used to control the learning
rate and thinning is applied to accept only a fixed number of parameters.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Create pSGLD sampler (with RMSProp preconditioner)
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
```

<!-- #region pycharm={"name": "#%% md\n"} -->
The sampler can now be used to sample parameters.
We provide the number of iterations and run the MCMC sampling algorithm.
We take the first (and only) chain indexed by `[0]` and from this we obtain the
sampled variables.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# Perform sampling
results = sampler(sample, iterations=iterations)
results = results[0]['samples']['variables']
```

<!-- #region pycharm={"name": "#%% md\n"} -->
Now the obtained samples provide 20 neural networks which we wish to evaluate.
We define a function which performs the evaluation.
Within this function we use the `full_data_mapper` from _JaxSGMC_ to map over
the full dataset (either training, validation or test dataset - depending on the 
evaluation settings) in batches of 1000 images. 
To do this, we define also the `fetch_logits` function which takes the 
parameters (current sample) and applies a neural network with these parameters 
on a batch of data and returns the predictions (logits).
Then we collect the logits for the full dataset and for each neural network.

Then we proceed with an aggregation of the ensemble results.
Two common approaches are taken here: hard voting and soft voting 
(also known as hard and soft aggregation).
For hard voting the logits are used to predict a class and then the class 
predicted by the majority is taken as the ensemble prediction.
In the case of soft voting the logits are converted to probabilities and the
mean of all ensemble members is taken - the class with the highest mean
probability is then chosen as the ensemble prediction.
This is similar to [sklearn.ensemble.VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html).

Furthermore, we are interested in investigating if ensemble certainty correlates
with higher accuracy.
We calculate the ensemble certainty as a measure of agreement among the members;
we set the certainty to be the ration of the number of members in the majority 
and all members - if all members predict the same class then the certainty is 100%.
Here we calculate the accuracies where the certainty is >= 50%, 60%, 70%, 80%, 90% and 100%.

Finally, we observe how the ensemble predictions provide a gateway to UQ when
making individual predictions.
We take five images drawn at random from the dataset and plot the predicted
probabilities as box-plots. This gives an insight into the uncertainty in the
prediction.
<!-- #endregion -->

```python pycharm={"name": "#%%\n"}
# This function get the logits for a batch of images
def fetch_logits(batch, mask, carry, params=None):
  temp_logits = apply_mobilenet(params, None, batch, is_training=False)
  return temp_logits, carry + 1
  
# Helper function to evaluate the accuracy for predictions which are more
# certain than a given threshold 
def threshold_accuracy(hard_class_pred, certainty, labels, certainty_threshold):
  # Get the predictions with certainty above the threshold
  selected = onp.asarray(certainty >= certainty_threshold)
  sel_hard_pred = hard_class_pred[selected]
  sel_labels = onp.squeeze(labels[selected])
  return onp.sum(sel_hard_pred == sel_labels) / onp.sum(selected)

def evaluate_model(results, loader, evaluation="hard", dataset="training"):
  # The full_data_mapper can apply the fetch_logits function to get the logits
  # for all images in the test, validation or training set
  my_parameter_mapper, sth_to_remove = data.full_data_mapper(loader, 1, 1000)

  # Prepare arrays to store the logits obtained for different neural network
  # parameters 
  if dataset == "validation" or dataset == "test":
    logits_all = onp.empty(
      (accepted_samples, test_labels.shape[0] // 2, num_classes))
  else:
    logits_all = onp.empty(
      (accepted_samples, train_labels.shape[0], num_classes))
  
  # Go over sampled parameters (NNs)
  for j in range(accepted_samples):
    params = tree_map(lambda x: jnp.array(x[j]), results['w'])
    # Collect logits
    out, _ = my_parameter_mapper(
     partial(fetch_logits, params=params), 0, masking=True)
    logits_all[j, :, :] = out.reshape(-1, 10)

  # We now investigate the logits using the hard voting or soft voting approach.
  # Hard-voting: obtain predicted class labels from each model and use the most 
  # frequently predicted class
  if evaluation == "hard":
    class_predictions = onp.argmax(logits_all, axis=-1)
    hard_class_predictions = onp.apply_along_axis(
      lambda x: onp.bincount(x, minlength=10).argmax(), 0, class_predictions)
      
    if dataset == "validation":
      accuracy = sum(
        hard_class_predictions == onp.squeeze(test_labels[test_labels.shape[0] // 2:, :]))
      accuracy /= (test_labels.shape[0] // 2)
    elif dataset == "test":
      accuracy = sum(
        hard_class_predictions == onp.squeeze(test_labels[:test_labels.shape[0] // 2, :]))
      accuracy /= (test_labels.shape[0] // 2)
    else:
      accuracy = sum(
        hard_class_predictions == onp.squeeze(train_labels))
      accuracy /= float(train_labels.shape[0])

    # Calculating certainty (per image)
    certainty = onp.count_nonzero(
      class_predictions == hard_class_predictions, axis=0)
    certainty = certainty / accepted_samples

    # Evaluating accuracy when certainty is above a fixed threshold
    accuracy_over_certainty = []
    for threshold in onp.linspace(0.5, 1.0, 6):
      # Select the correct labels
      if dataset == "validation":
        labels = test_labels[test_labels.shape[0] // 2:, :]
      elif dataset == "test":
        labels = test_labels[:test_labels.shape[0] // 2, :]
      else:
        labels = train_labels
        
      # Calculate the thresholded accuracy for the correct labels and transform
      # into %
      accuracy_over_certainty.append(
        100 * threshold_accuracy(
          hard_class_predictions, certainty, labels, threshold)
      )
    
    # Print the statistics on hard voting
    mode = (
      (dataset == "training") * "Training" 
      + (dataset == "validation") * "Validation"
      + (dataset == "test") * "Test"
    )
    print(f"{mode} Hard-Voting Accuracy: {accuracy * 100 :.2f} %\n"
          f"\tHard-Voting Accuracy-Over-Certainty: {accuracy_over_certainty} %")

  # Soft-voting: obtain predicted probabilities from each model and use the mean
  # of these probabilities to pick a class. The logits are the log-probability
  # that a sample belongs to a class.
  probabilities = onp.exp(logits_all)
  mean_probabilities = onp.mean(probabilities, axis=0)
  soft_class_predictions = onp.argmax(mean_probabilities, axis=-1)
  
  # Select the correct labels and draw trial images from the respective sets
  if dataset == "validation":
    labels = test_labels[test_labels.shape[0] // 2:, :]
    random_samples = onp.random.randint(0, test_labels.shape[0] // 2 - 1, 5)
  elif dataset == "test":
    labels = test_labels[:test_labels.shape[0] // 2, :]
    random_samples = onp.random.randint(0, test_labels.shape[0] // 2 - 1, 5)
  else:
    labels = train_labels
    random_samples = onp.random.randint(0, train_labels.shape[0] - 1, 5)
  
  accuracy = sum(soft_class_predictions == onp.squeeze(labels))
  accuracy = accuracy / labels.shape[0]
  
  # Print the statistics on hard voting
  print(f"{mode} Soft-Voting Accuracy: {accuracy * 100 :.2f} %")

  # Plotting the five randomly chosen images
  fig, ax = plt.subplots(1, 5, figsize=(10.8, 5))
  for i in range(len(random_samples)):
    ax[i].boxplot(probabilities[:, random_samples[i], :])
    ax[i].set_title("Image " + str(random_samples[i]))
    ax[i].set_xticks(list(range(1, 11)), [str(i) for i in range(10)])
  fig.tight_layout(pad=0.2)
  plt.show()
```

We perform evaluation for the training, validation and test set separately.

```python
# Model evaluation
evaluate_model(results, train_loader, dataset="training")
evaluate_model(results, val_loader, dataset="validation")
evaluate_model(results, test_loader, dataset="test")
```
