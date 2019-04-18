#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        # TODO: Define a suitable model, by calling `super().__init__`
        # with appropriate inputs and outputs.
        #
        # Alternatively, if you prefer to use a `tf.keras.Sequential`,
        # replace the `Network` parent, call `super().__init__` at the beginning
        # of this constructor and add layers using `self.add`.
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
        layers = args.cnn.split(',')
        hidden = self._create_hidden_layers(layers, inputs)
        outputs = tf.keras.layers.Dense(CIFAR10.LABELS,
                                        activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        # TODO: After creating the model, call `self.compile` with appropriate arguments.

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def _create_hidden_layers(self, layers, x):
        end_of_block = False
        conv = False
        for layer in layers:
            parameters = layer.split('-')

            if 'R' in parameters[0]:
                parameters.pop(0)
                parameters[0] = parameters[0][1:]
                residual_input = x

            if parameters[-1][-1] == ']':
                parameters[-1] = parameters[-1][:-1]
                end_of_block = True
                if parameters[-1][-1] == ']':
                    conv = True
                    filters = int(parameters[-1][-4:-1])
                    parameters[-1] = parameters[-1][:-5]

            if parameters[0] == 'C':
                x = tf.keras.layers.Conv2D(filters=int(parameters[1]),
                                           kernel_size=int(parameters[2]),
                                           strides=int(parameters[3]),
                                           padding=parameters[4],
                                           activation='relu')(x)
            elif parameters[0] == 'CB':
                x = tf.keras.layers.Conv2D(filters=int(parameters[1]),
                                           kernel_size=int(parameters[2]),
                                           strides=int(parameters[3]),
                                           padding=parameters[4],
                                           use_bias=True)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                if not end_of_block:
                    x = tf.keras.layers.ReLU()(x)
            elif parameters[0] == 'M':
                x = tf.keras.layers.MaxPool2D(pool_size=int(parameters[1]),
                                              strides=int(parameters[2]))(x)
            elif parameters[0] == 'F':
                x = tf.keras.layers.Flatten()(x)
            elif parameters[0] == 'D':
                x = tf.keras.layers.Dense(int(parameters[1]),
                                          activation='relu')(x)
            elif parameters[0] == 'A':
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=int(parameters[1]),
                    strides=int(parameters[2]))(x)
            else:
                print(parameters)
                raise ValueError('Type of layer not supported.')

            if end_of_block:
                if conv:
                    residual_input = tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=1,
                        strides=1,
                        padding='same')(residual_input)
                    residual_input = tf.keras.layers.BatchNormalization()(
                        residual_input)
                    conv = False
                x = tf.keras.layers.Add()([x, residual_input])
                x = tf.keras.layers.ReLU()(x)
                end_of_block = False

        return x

    def train(self, cifar, args):
        self.fit(
            cifar.train.data["images"], cifar.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(cifar.dev.data["images"],
                             cifar.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--cnn", default=None, type=str,
                        help="CNN architecture.")
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
    cifar = CIFAR10()

    # Create the network and train
    network = Network(args)
    network.train(cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
