import unicodedata, re, os, io, time, pickle, math, random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import efficientnet.tfkeras as efn

from tensorflow.keras.mixed_precision import experimental as mixed_precision
from kaggle_datasets import KaggleDatasets
from tqdm.notebook import tqdm

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', TPU.master())
except ValueError:
    print('Running on GPU')
    TPU = None

if TPU:
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    strategy = tf.distribute.get_strategy()

REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

DEBUG = False

# image resolution
IMG_HEIGHT = 256
IMG_WIDTH = 448
N_CHANNELS = 3
# maximum InChI length is 200 to prevent too much padding
MAX_INCHI_LEN = 200

# batch sizes
BATCH_SIZE_BASE = 6 if DEBUG else (64 if TPU else 12)
BATCH_SIZE = BATCH_SIZE_BASE * REPLICAS
BATCH_SIZE_DEBUG = 2

# target data type, bfloat16 when using TPU to improve throughput
TARGET_DTYPE = tf.bfloat16 if TPU else tf.float32
 # minimal memory usage of labels
LABEL_DTYPE= tf.uint8

# 100K validation images are used
VAL_SIZE = int(1e3) if DEBUG else int(100e3)
VAL_STEPS = VAL_SIZE // BATCH_SIZE

# ImageNet mean and std to normalize training images accordingly
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

# Google Cloud Dataset path to training and validation images
GCS_DS_PATH = KaggleDatasets().get_gcs_path('molecular-translation-images-cleaned-tfrecords')

# Tensorflow AUTO flag, used in datasets
AUTO = tf.data.experimental.AUTOTUNE

# dictionary to translate a character to the integer encoding
with open('molecular-translation-images-cleaned-tfrecords/vocabulary_to_int.pkl', 'rb') as handle:
    vocabulary_to_int = pickle.load(handle)

# dictionary to decode an integer encoded character back to the character
with open('molecular-translation-images-cleaned-tfrecords/int_to_vocabulary.pkl', 'rb') as handle:
    int_to_vocabulary = pickle.load(handle)
    
# configure model
VOCAB_SIZE = len(vocabulary_to_int.values())
SEQ_LEN_OUT = MAX_INCHI_LEN
DECODER_DIM = 512
CHAR_EMBEDDING_DIM = 256
ATTENTION_UNITS = 256

print(f'VOCAB_SIZE: {VOCAB_SIZE}')
