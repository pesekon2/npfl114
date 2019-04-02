#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=None, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=None, type=int, help="Window size to use.")
parser.add_argument('--activation_hidden', default='relu', type=str,
                    help='Activation functions in hidden layers.')
parser.add_argument('--optimizer', default=None, type=str)
parser.add_argument("--dropout", default=0, type=float,
                    help="Dropout regularization.")
parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
parser.add_argument("--label_smoothing", default=0, type=float,
                    help="Label smoothing.")
parser.add_argument('--activation_end', default='sigmoid', type=str,
                    help="Activation function for the last layer.",
                    choices=['softmax', 'sigmoid'])
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

a = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
# Create logdir name
args.logdir = os.path.join("logs", "{}-{}".format(#"{}-{}-{}".format(
    os.path.basename(__file__),
    # datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    a
))

# Load data
uppercase_data = UppercaseData(args.window, args.alphabet_size)

# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.

model = tf.keras.Sequential()
model.add(tf.keras.layers.Lambda(
    lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet)),
    input_shape=[2 * args.window + 1], dtype=tf.int32))
model.add(tf.keras.layers.Flatten())
for hidden_layer in args.hidden_layers:
    model.add(tf.keras.layers.Dense(hidden_layer,
                                    activation=args.activation_hidden))
    model.add(tf.keras.layers.Dropout(args.dropout))
# model.add(tf.keras.layers.Dense(args.alphabet_size, activation=tf.nn.softmax))
model.add(tf.keras.layers.Dense(2, activation=args.activation_end))

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000,
                                             profile_batch=1)
tb_callback.on_train_end = lambda *_: None

model.fit(uppercase_data.train.data['windows'],
          uppercase_data.train.data['labels'],
          validation_data=(uppercase_data.dev.data['windows'],
                           uppercase_data.dev.data['labels']),
          batch_size=args.batch_size, epochs=args.epochs,
          callbacks=[tb_callback])

test_logs = model.evaluate(uppercase_data.dev.data['windows'],
                           uppercase_data.dev.data['labels'],
                           batch_size=args.batch_size)
tb_callback.on_epoch_end(1, dict(('val_test_' + metric, value) for metric, value in zip(model.metrics_names, test_logs)))

model.save('uppercase_model_{}.h5'.format(a), include_optimizer=False)

accuracy = test_logs[model.metrics_names.index('accuracy')]

print(accuracy)

# model = tf.keras.models.load_model(
#     'uppercase_model_{}.h5'.format(a),
#     compile=False)

prediction = model.predict(uppercase_data.test.data['windows'])

with open("uppercase_test.txt", "w") as out_file:
    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.
    for i in range(len(uppercase_data.test.data['windows'])):# - 3):
        # letter = uppercase_data.test.data['windows'][i][3]
        if prediction[i][0] < prediction[i][1]:
            out_file.write(uppercase_data.test.text[i].capitalize())
        else:
            out_file.write(uppercase_data.test.text[i])

