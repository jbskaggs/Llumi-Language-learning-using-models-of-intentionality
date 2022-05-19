# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.configuration.deep_step import *
import tensorflow as tf
from src.old.training_data_language_v2 import *
import time

# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
# print(tf.config.list_physical_devices('GPU'))

if __name__ == "__main__":
    @tf.function
    def autoencoder_test_step(inputs, target, mask):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, token_ids, _ = pretrain_model(inputs, target, mask=mask, training=False)
        # loss = autoencoder_loss_object(inputs, predictions)
        return token_ids, predictions

    pretrain_model.load_weights(PRETRAIN_TENSOR_MODEL)

    # stamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    # logdir = '../src/model/logdir/' + stamp
    # writer = tf.summary.create_file_writer(logdir)

    epoch, i = 0, 0
    start = time.time()
    while True:
        zs = [input("input: ")]
        zs = training_data.convert_to_tokens(zs)
        mask = np.zeros((1, word_buffer_len))
        for j in range(len(zs)):
            if zs[0, j] == 103:
                mask[0, j] = 1
        tokens, prediction = autoencoder_test_step(zs, zs, mask=np.zeros_like(zs))
        prediction = np.array(prediction)
        prediction[:, :, 1] = 0
        prediction = np.argmax(prediction[0], axis=1)

        print('Prediction: ', training_data.convert_to_words(tokens[0]))
        print()


