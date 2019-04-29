#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# Dataset for generating sequences, with labels predicting whether the cumulative sum
# is odd/even.
class Dataset:
    def __init__(self, sequences_num, sequence_length, sequence_dim, seed,
                 shuffle_batches=True):
        sequences = np.zeros([sequences_num, sequence_length, sequence_dim],
                             np.int32)
        labels = np.zeros([sequences_num, sequence_length, 1], np.bool)
        generator = np.random.RandomState(seed)
        for i in range(sequences_num):
            sequences[i, :, 0] = generator.random_integers(
                0, max(1, sequence_dim - 1), size=[sequence_length])
            labels[i, :, 0] = np.bitwise_and(np.cumsum(sequences[i, :, 0]), 1)
            if sequence_dim > 1:
                sequences[i] = np.eye(sequence_dim)[sequences[i, :, 0]]
        self._data = {"sequences": sequences.astype(np.float32),
                      "labels": labels}
        self._size = sequences_num

        self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._size

    def batches(self, size=None):
        permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
        while len(permutation):
            batch_size = min(size or np.inf, len(permutation))
            batch_perm = permutation[:batch_size]
            permutation = permutation[batch_size:]

            batch = {}
            for key in self._data:
                batch[key] = self._data[key][batch_perm]
            yield batch


class Network:
    def __init__(self, args):
        sequences = tf.keras.layers.Input(shape=[args.sequence_length, args.sequence_dim])
        # TODO: Process the sequence using a RNN with cell type `args.rnn_cell`
        # and with dimensionality `args.rnn_cell_dim`. Use `return_sequences=True`
        # to get outputs for all sequence elements.
        #
        # Prefer `tf.keras.layers.LSTM` (and analogously for `GRU` and
        # `SimpleRNN`) to `tf.keras.layers.RNN` wrapper with
        # `tf.keras.layers.LSTMCell` (the former can run transparently on a GPU
        # and is also considerably faster on a CPU).

        predictions = self._build(sequences, args.rnn_cell, args.rnn_cell_dim,
                                  args.hidden_layer)

        # TODO: If `args.hidden_layer` is defined, process the result using
        # a ReLU-activated fully connected layer with `args.hidden_layer` units.

        # TODO: Generate predictions using a fully connected layer
        # with one output and `tf.nn.sigmoid` activation.
        self.model = tf.keras.Model(inputs=sequences, outputs=predictions)
        # super().__init__(inputs=sequences, outputs=predictions)

        # TODO: Create an Adam optimizer in self._optimizer
        # TODO: Create a suitable loss in self._loss
        # TODO: Create two metrics in self._metrics dictionary:
        #  - "loss", which is tf.metrics.Mean()
        #  - "accuracy", which is suitable accuracy

        self._optimizer = tf.keras.optimizers.Adam()
        self._loss = tf.keras.losses.BinaryCrossentropy()
        self._metrics = {'loss': tf.metrics.Mean(),
                         'accuracy': tf.metrics.BinaryAccuracy()}

        # TODO: Create a summary file writer using `tf.summary.create_file_writer`.
        # I usually add `flush_millis=10 * 1000` arguments to get the results reasonably quickly.
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10000)

    def _build(self, input, rnn_cell, dimension, hidden_layer):
        if rnn_cell == 'LSTM':
            x = tf.keras.layers.LSTM(units=dimension,
                                    return_sequences=True)(input)
        elif rnn_cell == 'GRU':
            x = tf.keras.layers.GRU(units=dimension,
                                   return_sequences=True)(input)
        elif rnn_cell == 'SimpleRNN':
            x = tf.keras.layers.SimpleRNN(units=dimension,
                                          return_sequences=True)(input)
        else:
            raise AttributeError('RNN cell type not supported.')

        if hidden_layer:
            x = tf.keras.layers.Dense(int(hidden_layer), activation='relu')(x)

        pred = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

        return pred

    @tf.function
    def train_batch(self, batch, clip_gradient):
        # TODO: Using a gradient tape, compute
        # - probabilities from self.model, passing `training=True` to the model
        # - loss, using `self._loss`
        # Then, compute `gradients` using `tape.gradients` with the loss and model variables.
        #
        # If clip_gradient is defined, clip the gradient and compute `gradient_norm` using
        # `tf.clip_by_global_norm`. Otherwise, only compute the `gradient_norm` using
        # `tf.linalg.global_norm`.
        #
        # Apply the gradients using the `self._optimizer`

        with tf.GradientTape() as tape:
            probs = self.predict_batch(batch, training=True)
            loss = self._loss(batch["labels"], probs)
            gradient = tape.gradient(loss, self.model.variables)
            if clip_gradient:
                gradient, gradient_norm = tf.clip_by_global_norm(gradient,
                                                                 clip_gradient)
            else:
                gradient_norm = tf.linalg.global_norm(gradient)

            self._optimizer.apply_gradients(zip(gradient, self.model.variables))

        # Generate the summaries. Start by setting the current summary step using
        # `tf.summary.experimental.set_step(self._optimizer.iterations)`.
        # Then, use `with self._writer.as_default():` block and in the block
        # - iterate through the self._metrics
        #   - reset each metric
        #   - for "loss" metric, apply currently computed `loss`
        #   - for other metrics, compute their value using the gold labels and predictions
        #   - then, add a summary using `tf.summary.scalar("train/" + name, metric.result())`
        # - Finall, add the gradient_norm using `tf.summary.scalar("train/gradient_norm", gradient_norm)`

            tf.summary.experimental.set_step(self._optimizer.iterations)
            with self._writer.as_default():
                for name, metric in self._metrics.items():
                    metric.reset_states()
                    if name == 'loss':
                        metric.apply(loss)
                    else:
                        metric.update_state(batch['labels'], probs)
                    tf.summary.scalar("train/" + name, metric.result())
                tf.summary.scalar("train/gradient_norm", gradient_norm)


    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            self.train_batch(batch, args.clip_gradient)

    @tf.function
    def predict_batch(self, batch, training=False):
        return self.model(batch["sequences"], training=training)

    def evaluate(self, dataset, args):
        # TODO: Similarly to training summaries, compute the metrics
        # averaged over all `dataset`.
        #
        # Start by resetting all metrics in `self._metrics`.
        #
        # Then iterate over all batches in `dataset.batches(args.batch_size)`.
        # - For each, predict probabilities using `self.predict_batch`.
        # - Compute loss of the batch
        # - Update the metrics (the "loss" metric uses current loss, other are computed
        #   using the gold labels and the predictions)
        #
        # Finally, create a dictionary `metrics` with results, using names and values in `self._metrics`.
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                # tf.summary.scalar("test/" + name, metric)
                metric.reset_states()

            for batch in dataset.batches(args.batch_size):
                probs = self.predict_batch(batch)
                loss = self._loss(batch['labels'], probs)
                self._metrics['loss'].apply(loss)
                self._metrics['accuracy'].update_state(batch['labels'], probs)

            metrics = {'loss': self._metrics['loss'].result(),
                       'accuracy': self._metrics['accuracy'].result()}

        return metrics

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--clip_gradient", default=None, type=lambda x: None if x == "None" else float(x), help="Gradient clipping norm.")
    parser.add_argument("--hidden_layer", default=None, type=lambda x: None if x == "None" else int(x), help="Dense layer after RNN.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
    parser.add_argument("--sequence_dim", default=1, type=int, help="Sequence element dimension.")
    parser.add_argument("--sequence_length", default=50, type=int, help="Sequence length.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--test_sequences", default=1000, type=int, help="Number of testing sequences.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train_sequences", default=10000, type=int, help="Number of training sequences.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
        tf.keras.utils.get_custom_objects()["orthogonal"] = lambda: tf.keras.initializers.orthogonal(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Create the data
    train = Dataset(args.train_sequences, args.sequence_length, args.sequence_dim, seed=42, shuffle_batches=True)
    test = Dataset(args.test_sequences, args.sequence_length, args.sequence_dim, seed=43, shuffle_batches=False)

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        network.train_epoch(train, args)
        metrics = network.evaluate(test, args)
    with open("sequence_classification.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["accuracy"]), file=out_file)