import random as rand
from random import random
import numpy as np
import math
import tensorflow as tf

# Parameters used.
MODEL_PATH = 'model.ckpt'


# Create a sequence classification instance.
def get_sequence(sequence_length):
    # Create a sequence of random numbers in [0,1].
    X = np.array([random() for _ in range(sequence_length)])
    # Calculate cut-off value to change class values.
    limit = sequence_length / 4.0
    # Determine the class outcome for each item in cumulative sequence.
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])

    return X, y


# Create n examples with random sequence lengths between 5 and 15.
def get_examples(n):
    X_list = []
    y_list = []
    sequence_length_list = []
    for _ in range(n):
        sequence_length = rand.randrange(start=5, stop=15)
        X, y = get_sequence(sequence_length)
        X_list.append(X)
        y_list.append(y)
        sequence_length_list.append(sequence_length)

    return X_list, y_list, sequence_length_list


# Tensorflow requires that all sentences (and all labels) inside the same batch have the same length,
# so we have to pad the data (and labels) inside the batches (with 0's, for example).
def pad(sentence, max_length):
    pad_len = max_length - len(sentence)
    padding = np.zeros(pad_len)
    return np.concatenate((sentence, padding))


# Create input batches.
def batch(data, labels, sequence_lengths, batch_size, input_size):
    n_batch = int(math.ceil(len(data) / batch_size))
    index = 0
    for _ in range(n_batch):
        batch_sequence_lengths = np.array(sequence_lengths[index: index + batch_size])
        batch_length = np.array(max(batch_sequence_lengths))  # max length in batch
        batch_data = np.array([pad(x, batch_length) for x in data[index: index + batch_size]])  # pad data
        batch_labels = np.array([pad(x, batch_length) for x in labels[index: index + batch_size]])  # pad labels
        index += batch_size

        # Reshape input data to be suitable for LSTMs.
        batch_data = batch_data.reshape(-1, batch_length, input_size)

        yield batch_data, batch_labels, batch_length, batch_sequence_lengths

# Generate train and test data.
x_train, y_train, sequence_length_train = get_examples(100)
x_test, y_test, sequence_length_test = get_examples(30)

