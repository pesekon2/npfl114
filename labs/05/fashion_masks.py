#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from fashion_masks_data import FashionMasks

# TODO: Define a suitable model in the Network class.
# A suitable starting model contains some number of shared
# convolutional layers, followed by two heads, one predicting
# the label and the other one the masks.
class Network(tf.keras.Model):
    def __init__(self, args):
        inputs, outputs = self._build()

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=[tf.keras.losses.SparseCategoricalCrossentropy(),
                           tf.keras.losses.BinaryCrossentropy()],
                     metrics=['accuracy'])
                     # metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
                     #            name='accuracy'),
                     #          tf.keras.metrics.SparseCategoricalAccuracy(
                     #            name='accuracy')])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, fashion_masks, args):
        for epoch in range(args.epochs):
            for x, y in self._prepare_batch(
                    fashion_masks.train.batches(args.batch_size)):
                self.train_on_batch(x, y)

            print("Dev {}: label: {}, mask: {}, both: {}".format(epoch + 1,
                                                                 *self.evaluate(
                                                                     fashion_masks.dev,
                                                                     args)))
        # self.fit(fashion_masks.train.data['images'],
        #          [fashion_masks.train.data['labels'],
        #           fashion_masks.train.data['masks']],
        #          batch_size=args.batch_size, epochs=args.epochs,
        #          validation_data=(fashion_masks.dev.data['images'],
        #                           [fashion_masks.dev.data['labels'],
        #                            fashion_masks.dev.data['masks']]),
        #          callbacks=[self.tb_callback])

    @staticmethod
    def _prepare_batch(batches_generator):
        for batch in batches_generator:
            model_inputs = batch["images"]
            model_targets = [
                batch["labels"],
                np.array([np.asarray(x).reshape(-1) for x in batch["masks"]])]
            yield (model_inputs, model_targets)
        # batches.clear()

    def evaluate(self, dataset, args):
        labels = list()
        masks = list()
        for inputs, targets in self._prepare_batch(
                dataset.batches(args.batch_size)):
            classification, mask = self.predict_on_batch(inputs)
            label = np.argmax(classification, axis=1)
            mask = np.array(np.round(mask), dtype=bool)
            labels.append(np.sum(label == targets[0]) / len(label))
            masks.append(np.sum(mask == targets[1], axis=1) / (28 * 28))

        return np.mean(labels), np.mean(masks), np.mean(labels) * np.mean(masks)

    def _build(self):
        input = tf.keras.layers.Input(shape=(FashionMasks.H, FashionMasks.W,
                                             FashionMasks.C))
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                                   padding='same', use_bias=True)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        c1 = x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                   padding='same', use_bias=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                   padding='same', use_bias=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, c1])
        c2 = x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                   padding='same', use_bias=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                   padding='same', use_bias=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, c2])
        c3 = x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                   padding='same', use_bias=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                   padding='same', use_bias=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        #
        c3_128 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1,
                                        padding='same')(c3)
        c3_128 = tf.keras.layers.BatchNormalization()(c3_128)
        #
        x = tf.keras.layers.Add()([x, c3_128])
        c4 = x = tf.keras.layers.ReLU()(x)
        av = tf.keras.layers.AveragePooling2D(
            pool_size=6, strides=1)(x)
        # TODO: WHY NOT 7????????????????

        class_head = tf.keras.layers.Flatten()(av)
        class_head = tf.keras.layers.Dense(1000,
                                           activation='relu')(class_head)
        class_head = tf.keras.layers.Dense(FashionMasks.LABELS,
                                           activation='softmax')(class_head)

        mask_head = tf.keras.layers.Flatten()(av)
        mask_head = tf.keras.layers.Dense(1000,
                                          activation='relu')(mask_head)
        mask_head = tf.keras.layers.Dense(FashionMasks.H * FashionMasks.W,
                                          activation='sigmoid')(mask_head)

        return input, [class_head, mask_head]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
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

    # Load data
    fashion_masks = FashionMasks()

    # Create the network and train
    network = Network(args)
    network.train(fashion_masks, args)
    # network.evaluate(fashion_masks.test, args)

    # Predict test data in args.logdir
    with open(os.path.join('logs', "fashion_masks_test.txt"), "w", encoding="utf-8") as out_file:
        # TODO: Predict labels and masks on fashion_masks.test.data["images"],
        # into test_labels and test_masks (test_masks is assumed to be
        # a Numpy array with values 0/1).
        test_labels, test_masks = network.predict(
            fashion_masks.test.data['images'], batch_size=args.batch_size)
        for label, mask in zip(test_labels, test_masks):
            print(label, *mask.astype(np.uint8).flatten(), file=out_file)

