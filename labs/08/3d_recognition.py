#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from modelnet import ModelNet

# The neural network model
class Network:
    def __init__(self, modelnet, args):
        # TODO: Define a suitable model, and either `.compile` it, or prepare
        # optimizer and loss manually.
        input, output = self._build(args)

        self.model = tf.keras.Model(inputs=input, outputs=output)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy")])

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def _build(self, modelnet, args):
        input = tf.layers.Input(shape=(args.modelnet, args.modelnet,
                                       args.modelnet))  # , 1?

        x = tf.keras.layers.Conv3D(filters=64, kernel_size=7, strides=2,
                                   padding='same', use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        c1 = x = tf.keras.layers.MaxPool3D(pool_size=3, strides=2)(x)

        x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, c1])
        c2 = x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, c2])
        c3 = x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        #
        c3_128 = tf.keras.layers.Conv3D(filters=128, kernel_size=1, strides=1,
                                        padding='same', use_bias=False)(c3)
        c3_128 = tf.keras.layers.BatchNormalization()(c3_128)
        #
        x = tf.keras.layers.Add()([x, c3_128])
        c4 = x = tf.keras.layers.ReLU()(x)

        av = tf.keras.layers.AveragePooling2D(pool_size=7, strides=1)(x)

        output = tf.keras.layers.Flatten()(av)
        output = tf.keras.layers.Dense(1000, activation='relu')(output)
        output = tf.keras.layers.Dense(len(modelnet.LABELS),
                                       activation='softmax')(output)

        return input, output

    def train(self, modelnet, args):
        # TODO: Train the network on a given dataset.
        self.model.fit(modelnet.train.data['voxels'],
                       modelnet.train.data['labels'],
                       batch_size=args.batch_size,
                       epochs=args.epochs,
                       validation_data=(modelnet.dev.data['voxels'],
                                        modelnet.dev.data['labels']),
                       callbacks=[self.tb_callback])
        # raise NotImplementedError()

    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndarray of
        # label probabilities from the test set
        return self.model.predict(dataset.data["voxels"])
        # raise NotImplementedError()


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    # tf.random.set_seed(42)
    # tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    # tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    modelnet = ModelNet(args.modelnet)

    # Create the network and train
  #  network = Network(modelnet, args)
  #  network.train(modelnet, args)

  #  # Generate test set annotations, but in args.logdir to allow parallel execution.
  #  out_path = "3d_recognition_test.txt"
  #  if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
  #  with open(out_path, "w", encoding="utf-8") as out_file:
  #      for probs in network.predict(modelnet.test, args):
  #          print(np.argmax(probs), file=out_file)
