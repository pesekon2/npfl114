#!/usr/bin/env py
import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, args, num_words, num_tags, num_chars):
        input, output = self._build(args, num_words, num_tags, num_chars)

        self.model = tf.keras.Model(inputs=input, outputs=output)

        self._optimizer = tf.optimizers.Adam()
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._metrics = {'loss': tf.metrics.Mean(), 'accuracy': tf.metrics.SparseCategoricalAccuracy()}
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def _build(self, args, num_words, num_tags, num_chars):
        word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        charseq_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        charseqs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)

        embedded_chars = tf.keras.layers.Embedding(input_dim=num_chars,
                                                   output_dim=32,
                                                   mask_zero=True)(charseqs)
        gru_chars = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(32, return_sequences=False),
            kernel_regularizer=tf.keras.regularizers.L1L2(l2=0.01),
            merge_mode="sum")(embedded_chars)
        replace = tf.keras.layers.Lambda(lambda args: tf.gather(*args))(
            [gru_chars, charseq_ids])
        embedded_words = tf.keras.layers.Embedding(input_dim=num_words,
                                                   output_dim=64,
                                                   mask_zero=True)(word_ids)
        concat = tf.keras.layers.Concatenate()([embedded_words, replace])
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            128, return_sequences=True), merge_mode="sum")(concat)
        hidden = tf.keras.layers.Dense(80,"relu")(lstm)
        preds = tf.keras.layers.Dense(num_tags, activation="softmax")(lstm)

        return [word_ids, charseq_ids, charseqs], preds

    def evaluate(self, dataset, args):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            self.evaluate_batch(
                [batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs],
                batch[dataset.TAGS].word_ids)

        metrics = {name: metric.result() for name, metric in self._metrics.items()}
        return metrics

    def evaluate_batch(self, inputs, tags):
        tags_ex = np.expand_dims(new_tags, axis=2)
        mask = tf.not_equal(tags_ex, 0)
        probabilities = self.model(inputs, training=False)
        loss = self._loss(tags_ex, probabilities, mask)
        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                metric(tags_ex, probabilities, mask)

    def gen_tags(self,tags):
        longer_tags = [self.tag_dict[tuple(tag)] for tag in tags]
        stacked = np.vstack(longer_tags)
        return stacked

    def train_batch(self, inputs, tags):
        tags_ex = np.expand_dims(tags, axis=2)
        mask = tf.not_equal(tags_ex, 0)

        with tf.GradientTape() as tape:
            probabilities = self.model(inputs, training=True)
            loss = self._loss(tags_ex, probabilities, mask)
        gradients = tape.gradient(loss, self.model.variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric(loss)
                else:
                    metric(tags_ex, probabilities, mask)


    def train(self, train_data, dev_data, args):
        for epoch in range(0,args.epochs):
            for batch in train_data.batches(args.batch_size):
                self.train_batch([batch[train_data.FORMS].word_ids,
                                  batch[train_data.FORMS].charseq_ids,
                                  batch[train_data.FORMS].charseqs],
                                 batch[train_data.TAGS].word_ids)
            # Evaluate on dev data
            metrics = network.evaluate(dev_data, args)
            print("Dev accuracy: ", metrics['accuracy'])

    def predict(self, dataset, args):
        # TODO: Predict method should return a list, each element corresponding
        # to one sentence. Each sentence should be a list/np.ndarray
        # containing _indices_ of chosen tags (not the logits/probabilities).
        predictions = self.model([dataset.data[dataset.FORMS].word_ids,
                                  dataset.data[dataset.FORMS].charseq_ids,
                                  dataset.data[dataset.FORMS].charseqs],training=False)
        edited = tf.argmax(predictions, axis=2)
        return edited

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt", max_sentences=1000)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Create the network and train
    network = Network(args, num_words=len(morpho.train.data[morpho.train.FORMS].words),
                            num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                            num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    network.train(morpho.train, morpho.dev, args)
    p = network.predict(morpho.test, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "tagger_competition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(network.predict(morpho.test, args)):
            for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
                print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
                      morpho.test.data[morpho.test.LEMMAS].word_strings[i][j],
                      morpho.test.data[morpho.test.TAGS].words[sentence[j]],
                      sep="\t", file=out_file)
            print(file=out_file)
