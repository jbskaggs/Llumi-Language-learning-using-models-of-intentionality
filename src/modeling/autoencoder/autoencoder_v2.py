# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.configuration.deep_step import *
from src.dataset_processing.training_data import *
import time

if __name__ == "__main__":
    # # Use this to start at an article other than 0
    # training_data.current_article_id = 72995
    # training_data.get_new_article()
    # pretrain_model.load_weights(PRETRAIN_TENSOR_MODEL)

    EXAMPLES_PER_EPOCH_WIKIPEDIA = 1000
    EXAMPLES_PER_EPOCH_MOVIE = 100
    EXAMPLES_PER_EPOCH_SSHARP = 1  # 5

    epoch = 0
    start = time.time()
    while training_data.tdw.current_article_id != 0:
        # wikipedia dataset
        for i in range(EXAMPLES_PER_EPOCH_WIKIPEDIA):
            zs_out = training_data.get_batch('wikipedia')
            zs_inp, zs_out, mask = training_data.get_masked_input(zs_out)
            results, vocab_loss = autoencoder_train_step(zs_inp, zs_out, mask)
            results = np.array(results)
            vocab_loss = np.array(vocab_loss)

        # movie dataset
        for i in range(EXAMPLES_PER_EPOCH_MOVIE):
            zs_out, _, _ = training_data.get_batch('movie')
            zs_inp, zs_out, mask = training_data.get_masked_input(zs_out)
            results, vocab_loss = autoencoder_train_step(zs_inp, zs_out, mask)
            results = np.array(results)
            vocab_loss = np.array(vocab_loss)

        # ssharp dataset
        for i in range(EXAMPLES_PER_EPOCH_SSHARP):
            training_data.load_next_batch()
            zs_out, _, _, _ = training_data.get_batch('ssharp')
            zs_inp, zs_out, mask = training_data.get_masked_input(zs_out)
            results, vocab_loss = autoencoder_train_step(zs_inp, zs_out, mask)
            results = np.array(results)
            vocab_loss = np.array(vocab_loss)

        zs_out = training_data.get_batch('wikipedia')
        zs_inp, zs_out, mask = training_data.get_masked_input(zs_out)
        results, vocab_loss = autoencoder_train_step(zs_inp, zs_out, mask)
        print()
        print(
            f'Epoch: {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Runtime: {time.time() - start}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Article Number: {training_data.tdw.current_article_id}, '
        )
        print('Original  : ', training_data.convert_to_words(zs_out[0]))
        print('Masked    : ', training_data.convert_to_words(zs_inp[0]))
        print('Prediction:  [CLS]', training_data.convert_to_words(results[0]))
        print()

        pretrain_model.save_weights(PRETRAIN_TENSOR_MODEL)

        epoch += 1
        train_loss.reset_states()
        train_accuracy.reset_states()
        start = time.time()
