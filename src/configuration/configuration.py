import tensorflow as tf
from src.dataset_processing.training_data import *
import sys
from tensorflow.keras import mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

debug_mode = sys.gettrace() is not None
if debug_mode:
    # run everything on the cpu
    tf.config.run_functions_eagerly(True)
else:
    # run normally on the gpu
    pass

# # import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')`
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


autoencoder_bool = False

# tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float32')

# vocab = open('vocab.csv').read().split(',')
# TENSOR_MODEL = "..\\tensor_model\\model.ckpt"
# TENSOR_MODEL_2 = "..\\tensor_model_2\\model.ckpt"
PRETRAIN_TENSOR_MODEL = '../../model/saved_model/pretrain_model'
STATE_TENSOR_MODEL = '../../model/saved_model/state_model'
MIDTRAIN_TENSOR_MODEL = '../../model/saved_model/midtrain_model'
TRAIN_TENSOR_MODEL = '../../model/saved_model/train_model'
EVENTS = '../events'

# model architecture
num_layers = 12
num_heads = 8
embedding_size = 512

# model hyper-parameters we ran .4 epochs
learning_rate = .00001  # original=.00001
# autoencoder_batch_size = 30

batch_size = 25
word_buffer_len = 40
chitchat = False

# dont touch these
# state_embedding_size = 204
state_embedding_size = 40

if chitchat:
    ssharp_data = 'ssharp_chitchat'
else:
    ssharp_data = 'ssharp'

# set-up training data and vocabulary
# training_data = TrainingData(batch_size, word_buffer_len, ['wikipedia', 'movie', ssharp_data])
training_data = TrainingData(batch_size, word_buffer_len, [ssharp_data])
vocab = training_data.get_vocabulary()
vocab_size = len(vocab)




