# import os
# import cv2
import numpy as np
# from glob import glob
# from scipy.io import loadmat
# import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Union

from src.models_arch import *

import time
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.metrics import Metric
from keras import layers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.layers import *
import keras
from tensorflow.keras.optimizers import *

from keras import backend as K
from keras.backend import binary_crossentropy

from typing import Callable, Dict, List, Optional, Union
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Model
import tensorflow as tf


####################################################################################################


####################################################################################################


# class BinaryMeanIoU(tf.keras.metrics.Metric):
#     def __init__(self, name='binary_mean_iou', threshold=0.5, **kwargs):
#         super(BinaryMeanIoU, self).__init__(name=name, **kwargs)
#         self.threshold = threshold
#         self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)
    
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.cast(y_pred > self.threshold, tf.float32)
#         return self.mean_iou.update_state(y_true, y_pred, sample_weight)
    
#     def result(self):
#         return self.mean_iou.result()
    
#     def reset_state(self):
#         self.mean_iou.reset_states()
        

####################################################################################################

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true = tf.cast(y_true, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# --------------------------------------------------------


def jaccard_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1) - intersection + 1e-6
    jaccard_loss = 1 - (intersection / union)
    return jaccard_loss

def jaccard_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])  # Sum the product in all axes
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])  # Sum the sum in all axes
    jac = (intersection + smooth) / (sum_ - intersection + smooth)  # Calc jaccard
    return K.mean(jac)

def jaccard_coef_thresholded(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))  # Round to 0 or 1
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def jaccard_coef_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

# ------------------------------------------------------

# This is a loss function
def jacc_coef(y_true, y_pred):
    smooth = 0.0000001
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

####################################################################################################

# def dice_loss(y_true, y_pred):
#     numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
#     denominator = tf.reduce_sum(y_true + y_pred, axis=-1) + 1e-6
#     dice_loss = 1 - (numerator / denominator)
#     return dice_loss

# def jaccard_loss(y_true, y_pred):
#     intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
#     union = tf.reduce_sum(y_true + y_pred, axis=-1) - intersection + 1e-6
#     jaccard_loss = 1 - (intersection / union)
#     return jaccard_loss


# def jaccard_coef(y_true, y_pred):
#     """
#     Calculates the Jaccard index
#     """
#     # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
#     smooth = 1e-12
#     intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])  # Sum the product in all axes

#     sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])  # Sum the sum in all axes

#     jac = (intersection + smooth) / (sum_ - intersection + smooth)  # Calc jaccard

#     return K.mean(jac)


# def jaccard_coef_thresholded(y_true, y_pred):
#     """
#     Calculates the binarized Jaccard index
#     """
#     # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
#     smooth = 1e-12

#     # Round to 0 or 1
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))

#     # Calculate Jaccard index
#     intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
#     sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)

#     return K.mean(jac)


# def jaccard_coef_loss(y_true, y_pred):
#     """
#     Calculates the loss as a function of the Jaccard index and binary crossentropy
#     """
#     # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
#     return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


####################################################################################################


# Define the available models globally
AVAILABLE_MODELS = {
    'simple_net': mk_simple_net,
    'unet': mk_multi_unet_model,
    # 'unet_2': mk_DL_L8S2_UV_UNET,
    'unet_plus_plus': mk__unet_plusplus,
    'rs_net': mk_rs_net,
    'deep_lab_v3_plus': mk_DeeplabV3Plus,
    'CXNet': mk_cloudXnet,
    'cloud_net': mk_cloud_net,
    # 'GAN': None,
    }


def create_model(
    model_name: str = 'simple_net', 
    n_classes: int = 1, 
    IMG_HEIGHT: Optional[int] = None, 
    IMG_WIDTH: Optional[int] = None, 
    IMG_CHANNELS: int = 4,
    ) -> Model:

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid model name '{model_name}'. "
            f"Available models are: {', '.join(AVAILABLE_MODELS.keys())}"
        )

    build_model = AVAILABLE_MODELS[model_name]
    
    if build_model is None:
        raise NotImplementedError(f"Model '{model_name}' is not implemented yet.")
    
    model = build_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    return model


def compile_model(
    model: Model,
    optimizer: Union[str, Callable] = 'adam', 
    loss: Union[str, Callable] = 'binary_crossentropy', 
    loss_params: Optional[Dict[str, float]] = None, 
    optimizer_params: Optional[Dict[str, float]] = None,
    metrics: Optional[Union[str, List[str], List[Metric]]] = None
) -> Model:
    
    optimizer = tf.keras.optimizers.get(optimizer)
    
    if optimizer_params is not None:
        optimizer = optimizer(**optimizer_params)
    
    loss = get_loss(loss, loss_params)
    
    metrics = get_metrics(metrics)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def get_loss(loss: Union[str, Callable], loss_params: Optional[Dict[str, float]] = None) -> Callable:
    if isinstance(loss, str):
        loss = tf.keras.losses.get(loss)

    if loss_params is not None:
        loss = loss(**loss_params)

    return loss


def get_metrics(metrics: Optional[Union[str, List[str], List[Metric]]]) -> List[Metric]:
    if metrics is None:
        return [tf.keras.metrics.AUC()]
    if isinstance(metrics, str):
        return [tf.keras.metrics.get(metrics)]
    if isinstance(metrics, list):
        return [tf.keras.metrics.get(m) if isinstance(m, str) else m for m in metrics]
    

####################################################################################################


