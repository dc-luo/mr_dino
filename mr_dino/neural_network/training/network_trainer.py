import pickle 
import time 

import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf

from .losses import normalized_mse, normalized_mse_matrix, \
        l2_accuracy, f_accuracy_matrix

from .network_jacobian import equip_model_with_full_control_jacobian

# if int(tf.__version__[0]) > 1:
    # import tensorflow.compat.v1 as tf
    # tf.disable_v2_behavior()

class SwitchScheduler:
    def __init__(self, scale=0.5, epoch_tol=1000):
        self.epoch_tol = epoch_tol
        self.scale=scale

    def __call__(self, epoch, lr):
        if epoch == self.epoch_tol:
            return self.scale * lr 
        else:
            return lr

class ExponentialDecayScheduler:
    def __init__(self, scale=0.9, epoch_tol=1000):
        self.epoch_tol = epoch_tol
        self.scale=scale

    def __call__(self, epoch, lr):
        if epoch < self.epoch_tol:
            return lr
        else:
            return self.scale * lr 



def train_network_mr_dino(network, input_data_with_jacobian, output_data_with_jacobian, 
        optimizer, epochs, batch_size, validation_data=None, scheduler=None, verbosity=0):
    """
    Equip network with jacobian and train
    - :code: `network` a Tensorflow model
    - :code: `input_data_with_jacobian` list of input data 
    - :code: `output_data_with_jacobian` list of output data with jacobian [function, jacobian]
    - :code: `optimizer` keras optimizer
    - :code: `epochs` number of epochs for training
    - :code: `batch_size` batch size for optimizer
    - :code: `validation_data` dictionary for validation data. Optional
    - :code: `scheduler` a learning rate scheduler type. Optional
    - :code: `verbosity` verbosity of training outputs in tf. Default = 0

    Returns the training time and the logged history
    """
    jacobian_network = equip_model_with_full_control_jacobian(network)
    loss = [normalized_mse, normalized_mse_matrix]
    metrics = [l2_accuracy, f_accuracy_matrix]
    jacobian_network.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    t0 = time.time()
    if scheduler is not None:
        callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
    else:
        callbacks = None

    history = jacobian_network.fit(input_data_with_jacobian, output_data_with_jacobian,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=verbosity)

    t1 = time.time()
    training_time = t1 - t0
    return training_time, history

def train_network_l2(network, input_data, output_data, optimizer, epochs, batch_size, 
        validation_data=None, scheduler=None, verbosity=0):
    """
    Train network with l2 misfit 
    - :code: `network` a Tensorflow model
    - :code: `input_data` list of input data 
    - :code: `output_data` list of output data 
    - :code: `optimizer` keras optimizer
    - :code: `epochs` number of epochs for training
    - :code: `batch_size` batch size for optimizer
    - :code: `validation_data` dictionary for validation data. Optional
    - :code: `scheduler` a learning rate scheduler type. Optional
    - :code: `verbosity` verbosity of training outputs in tf. Default = 0

    Returns the training time and the logged history
    """
    loss = normalized_mse
    metrics = [l2_accuracy]

    if scheduler is not None:
        callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
    else:
        callbacks = None

    network.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    t0 = time.time()
    history = network.fit(input_data, output_data,
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose = verbosity)
    t1 = time.time()
    training_time = t1 - t0
    return training_time, history



