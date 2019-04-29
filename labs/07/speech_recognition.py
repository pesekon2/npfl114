#!/usr/bin/env python3
import contextlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Embedding, Dense, Lambda, concatenate

from timit_mfcc import TimitMFCC

tf.config.gpu.set_per_process_memory_growth(True)


class Network:
    def __init__(self, args):
        self._beam_width = args.ctc_beam
        self.best_distance = 1.0

        # TODO: Define a suitable model, given already masked `mfcc` with shape
        # `[None, TimitMFCC.MFCC_DIM]`. The last layer should be a Dense layer
        # without an activation and with `len(TimitMFCC.LETTERS) + 1` outputs,
        # where the `+ 1` is for the CTC blank symbol.
        #
        # Store the results in `self.model`.

        input, predictions = self._build()

        self.model = tf.keras.Model(inputs=input, outputs=predictions)

        # The following are just defaults, do not hesitate to modify them.
        self._optimizer = tf.optimizers.Adam()
        self._metrics = {"loss": tf.metrics.Mean(), "edit_distance": tf.metrics.Mean()}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def _build(self):
        input = tf.keras.layers.Input(shape=(None, TimitMFCC.MFCC_DIM))
        rnn = Bidirectional(GRU(units=128,
                                return_sequences=True))(input)
        preds = Dense(units=len(TimitMFCC.LETTERS) + 1, activation=None)(rnn)

        return input, preds

    # Converts given tensor with `0` values for padding elements, create
    # a SparseTensor.
    def _to_sparse(self, tensor):
        tensor_indices = tf.where(tf.not_equal(tensor, 0))
        return tf.sparse.SparseTensor(tensor_indices, tf.gather_nd(tensor, tensor_indices), tf.shape(tensor, tf.int64))

    # Convert given sparse tensor to a (dense_output, sequence_lengths).
    def _to_dense(self, tensor):
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor_lens = tf.reduce_sum(tf.cast(tf.not_equal(tensor, -1), tf.int32), axis=1)
        return tensor, tensor_lens

    # Compute logits given input mfcc, mfcc_lens and training flags.
    # Also transpose the logits to `[time_steps, batch, dimension]` shape
    # which is required by the following CTC operations.
    def _compute_logits(self, mfcc, mfcc_lens, training):
        logits = self.model(mfcc, mask=tf.sequence_mask(mfcc_lens), training=training)
        return tf.transpose(logits, [1, 0, 2])

    # Compute CTC loss using given logits, their lengths, and sparse targets.
    def _ctc_loss(self, logits, logits_len, sparse_targets):
        loss = tf.nn.ctc_loss(sparse_targets, logits, None, logits_len, blank_index=len(TimitMFCC.LETTERS))
        self._metrics["loss"](loss)
        return tf.reduce_mean(loss)

    # Perform CTC predictions given logits and their lengths.
    def _ctc_predict(self, logits, logits_len):
        (predictions,), _ = tf.nn.ctc_beam_search_decoder(logits, logits_len, beam_width=self._beam_width)
        return tf.cast(predictions, tf.int32)

    # Compute edit distance given sparse predictions and sparse targets.
    def _edit_distance(self, sparse_predictions, sparse_targets):
        edit_distance = tf.edit_distance(sparse_predictions, sparse_targets, normalize=True)
        self._metrics["edit_distance"](edit_distance)
        return edit_distance

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, TimitMFCC.MFCC_DIM], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def train_batch(self, mfcc, mfcc_lens, targets):
        sparse_targets = self._to_sparse(tf.cast(targets, dtype=tf.int32))

        with tf.GradientTape() as tape:
            logits = self._compute_logits(mfcc, mfcc_lens, training=True)
            loss = self._ctc_loss(logits, mfcc_lens, sparse_targets)

        gradients = tape.gradient(loss, self.model.variables)

        gradients, _ = tf.clip_by_global_norm(gradients, 0.1)
        self._optimizer.apply_gradients(zip(gradients, self.model.variables))

        self._edit_distance(self._ctc_predict(logits, mfcc_lens), sparse_targets)

    def train_epoch(self, epoch, dataset, args):
        for _, metric in self._metrics.items():
            metric.reset_states()

        batches_num = dataset.size / args.batch_size

        for i, batch in enumerate(dataset.batches(args.batch_size)):
            self.train_batch(batch["mfcc"], batch["mfcc_len"], batch["letters"])

            print('\repoch: {} | train loss: {} | train edit dist: {} | {} %'.format(epoch, self._metrics["loss"].result(), self._metrics["edit_distance"].result(), int(i / batches_num * 100)), end='', flush=True)
        print('\repoch: {} | train loss: {} | train edit dist: {} | '.format(epoch, self._metrics["loss"].result(), self._metrics["edit_distance"].result()), end='', flush=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, TimitMFCC.MFCC_DIM], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def evaluate_batch(self, mfcc, mfcc_lens, targets):
        sparse_targets = self._to_sparse(targets)

        logits = self._compute_logits(mfcc, mfcc_lens, training=True)
        self._edit_distance(self._ctc_predict(logits, mfcc_lens), sparse_targets)

    def evaluate(self, dataset, dataset_name, args):
        for _, metric in self._metrics.items():
            metric.reset_states()

        for batch in dataset.batches(args.batch_size):
            self.evaluate_batch(batch["mfcc"], batch["mfcc_len"], batch["letters"])

        print('eval edit dist: {self._metrics["edit_distance"].result():.4f}', flush=True)

        if self._metrics["edit_distance"].result() < self.best_distance:
            self.best_distance = self._metrics["edit_distance"].result()
            self.model.save_weights(
                "/workspace/npfl114/labs/07/GRU-3-{128}_acc-{self.best_distance:.4f}")

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, TimitMFCC.MFCC_DIM], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def predict_batch(self, mfcc, mfcc_lens):
        # TODO: Implement batch prediction, returning a tuple (dense_predictions, prediction_lens)
        # produced by self._to_dense.
        logits = self._compute_logits(mfcc, mfcc_lens, False)
        predictions = self._ctc_predict(logits, mfcc_lens)
        dense = self._to_dense(predictions)

        return dense

    def predict(self, dataset, args):
        sentences = []
        for batch in dataset.batches(args.batch_size):
            for prediction, prediction_len in zip(*self.predict_batch(batch["mfcc"], batch["mfcc_len"])):
                sentences.append(prediction[:prediction_len])
        return sentences


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--ctc_beam", default=16, type=int, help="CTC beam.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    timit = TimitMFCC()

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        network.train_epoch(epoch, timit.train, args)
        network.evaluate(timit.dev, "dev", args)

        # Generate test set annotations, but to allow parallel execution, create it
        # in in args.logdir if it exists.
        out_path = "speech_recognition_test{}.txt".format(epoch)
        if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
        with open(out_path, "w", encoding="utf-8") as out_file:
            for sentence in network.predict(timit.test, args):
                print(" ".join(timit.LETTERS[letters] for letters in sentence), file=out_file)

