# -*- coding: utf-8 -*-
"""Simple Keras Integration
docs: https://docs.wandb.ai/integrations/keras

Original file is located at
    https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Simple_Keras_Integration.ipynb

## What this code does

We show you how to integrate Weights & Biases with your Keras code to add experiment tracking to your pipeline. That includes:

1. Storing hyperparameters and metadata in a `config`.
2. Passing `WandbCallback` to `model.fit`. This will automatically log training metrics, like loss, and system metrics, like GPU and CPU utilization.
3. Using the `wandb.log` API to log custom metrics.

all using the CIFAR-10 dataset.

Then, we'll show you how to catch your model making mistakes by logging both the output predictions and the input images the network used to generate them.

### Follow along with a [video tutorial](https://tiny.cc/wb-keras-video)!
**Note**: Sections starting with _Step_ are all you need to integrate W&B in an existing pipeline. The rest just loads data and defines a model.

# Install, Import, and Log In
"""

import os
import random
# import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import wandb
from wandb.keras import WandbCallback

# Set the random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = '1' 
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)

# Print the config
# https://docs.wandb.ai/library/environment-variables
github_sha = os.getenv('GITHUB_SHA')
entity = os.getenv('WANDB_ENTITY')
project = os.getenv('WANDB_PROJECT')

print(f'Entity: {entity}')
print(f'Project: {project}')

# Download and Prepare the Dataset

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Subsetting train data and normalizing to [0., 1.]
x_train, x_test = x_train[::5] / 255., x_test / 255.
y_train = y_train[::5]

CLASS_NAMES = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]

print('Shape of x_train: ', x_train.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of x_test: ', x_test.shape)    
print('Shape of y_test: ', y_test.shape)

"""# Define the Model

Here, we define a standard CNN (with convolution and max-pooling) in Keras.
"""

def Model():
  inputs = keras.layers.Input(shape=(32, 32, 3))

  x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
  x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size=2)(x)

  x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
  x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)

  x = keras.layers.GlobalAveragePooling2D()(x)

  x = keras.layers.Dense(128, activation='relu')(x)
  x = keras.layers.Dense(32, activation='relu')(x)
  
  outputs = keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

  return keras.models.Model(inputs=inputs, outputs=outputs)

"""# Train the Model

### Step 2: Give `wandb.init` your `config`

You first initialize your wandb run, letting us know some training is about to happen. [Check out the official documentation for `.init` here $\rightarrow$](https://docs.wandb.com/library/init)

That's when you need to set your hyperparameters.
They're passed in as a dictionary via the `config` argument,
and then become available as the `config` attribute of `wandb`.

Learn more about `config` in this [Colab Notebook $\rightarrow$](https://colab.research.google.com/drive/1UQ-lc9NmHOhyxWnLECefolxEiwB_Cxx3)
"""

# Initialize wandb with your project name
run = wandb.init(project='my-keras-integration',
                 config={  # and include hyperparameters and metadata
                     "learning_rate": 0.005,
                     "epochs": 5,
                     "batch_size": 64,
                     "loss_function": "sparse_categorical_crossentropy",
                     "architecture": "CNN",
                     "dataset": "CIFAR-10",
                     "github_sha": github_sha
                 })
config = wandb.config  # We'll use this to configure our experiment

# Initialize model like you usually do.
tf.keras.backend.clear_session()
model = Model()
model.summary()

# Compile model like you usually do.
# Notice that we use config, so our metadata matches what gets executed
optimizer = tf.keras.optimizers.Adam(config.learning_rate) 
model.compile(optimizer, config.loss_function, metrics=['acc'])

"""### Step 3: Pass `WandbCallback` to `model.fit`

Keras has a [robust callbacks system](https://keras.io/api/callbacks/) that
allows users to separate model definition and the core training logic
from other behaviors that occur during training and testing.

That includes, for example, 

**Click on the Project page link above to see your results!**
"""

# We train with our beloved model.fit
# Notice WandbCallback is used as a regular callback
# We again use config
_ = model.fit(x_train, y_train,
          epochs=config.epochs, 
          batch_size=config.batch_size,
          validation_data=(x_test, y_test),
          callbacks=[WandbCallback()])

"""# Use `wandb.log` for custom metrics

Here, we log the error rate on the test set.
"""

loss, accuracy = model.evaluate(x_test, y_test)
print('Test Error Rate: ', round((1 - accuracy) * 100, 2))

# With wandb.log, we can easily pass in metrics as key-value pairs.
wandb.log({'Test Error Rate': round((1 - accuracy) * 100, 2)})

run.join()
