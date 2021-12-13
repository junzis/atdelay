import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))