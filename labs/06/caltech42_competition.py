#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub
import cv2

from caltech42 import Caltech42

# The neural network model
class Network:
    def __init__(self, args, caltech42):
        # TODO: You should define `self.model`. You should use the following layer:
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280])
        # The layer:
        # - if given `trainable=True/False` to KerasLayer constructor, the layer weights
        #   either are marked or not marked as updatable by an optimizer;
        # - however, batch normalization regime is set independently, by `training=True/False`
        #   passed during layer execution.
        #
        # Therefore, to not train the layer at all, you should use
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)
        #   features = mobilenet(inputs, training=False)
        # On the other hand, to fully train it, you should use
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=True)
        #   features = mobilenet(inputs)
        # where the `training` argument to `mobilenet` is passed automatically in that case.
        #
        # Note that a model with KerasLayer can currently be saved only using
        #   tf.keras.experimental.export_saved_model(model, path, serving_only=True/False)
        # where `serving_only` controls whether only prediction, or also training/evaluation
        # graphs are saved. To again load the model, use
        #   model = tf.keras.experimental.load_from_saved_model(path, {"KerasLayer": tfhub.KerasLayer})

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir,
                                                        update_freq=1000,
                                                        profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

        inputs, outputs = self._build(args)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self._writer = tf.summary.create_file_writer(args.logdir,
                                                     flush_millis=10000)
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self._metrics = {
            'loss': tf.metrics.Mean(),
            'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy")}
        self._optimizer = tf.keras.optimizers.Adam()
        self.tb_callback.set_model(self.model)

        self.model.compile(
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=[self._metrics['accuracy']])

    def _build(self, args):
        inputs = tf.keras.layers.Input(shape=(caltech42.MIN_SIZE,
                                              caltech42.MIN_SIZE,
                                              caltech42.C))
        mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/"
                                     "mobilenet_v2/feature_vector/2",
                                     output_shape=[1280], trainable=False)
        features = mobilenet(inputs, training=True)
        features = tf.keras.layers.Dense(args.hidden_layer,
                                         activation='relu')(features)
        features = tf.keras.layers.Dense(caltech42.LABELS,
                                         activation='softmax')(features)

        return inputs, features

    def train(self, caltech42, args):
        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for epoch in range(args.epochs):
                for batch in caltech42.train.batches(size=args.batch_size):
                    self.model.train_on_batch(batch['images'], batch['labels'])

                # self.model.fit_generator(
                #     generator=caltech42.train.batches(args.batch_size),#, repeat=True),
                #     steps_per_epoch=caltech42.train.batched_size(args.batch_size),
                #     epochs=args.epochs,
                #     validation_data=caltech42.dev.batches(args.batch_size),#,
                #                                           #repeat=True),
                #     validation_steps=caltech42.dev.batched_size(args.batch_size),
                #     callbacks=[self.tb_callback])  # , checkpoint_callback]),
                pred = self.model(caltech42.dev.data['images'])
                label = caltech42.dev.data['labels']
                acc = self._metrics['accuracy'](label, pred)
                #########
                loss = self._loss(label, pred)
                # metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
                self.tb_callback.on_epoch_end(epoch, {'accuracy': acc})
                tf.summary.scalar("train/" + 'accuracy', acc)
                ###########

                print("Epoch: ", epoch,
                      "Accuracy: ", acc)

    def predict(self, caltech42, args):
        a = list()

        for batch in caltech42.batches(size=args.batch_size):
            probs = self.model(batch["images"])
            a.append(probs)

        a = tf.concat(a, 0)
        return a

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=70, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1500, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--hidden_layer", default=2000, type=int,
                        help="Depth of the dense layer.")
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
    os.makedirs(args.logdir)

    #########################################################
    def process_im(im):
        decoded = cv2.imdecode(np.frombuffer(im, np.uint8), -1)
        resized = np.array(cv2.resize(decoded, dsize=(224, 224), interpolation=cv2.INTER_CUBIC), dtype=np.float32)
        if resized.shape != (Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C):
            resized = cv2.cvtColor(resized,cv2.COLOR_GRAY2RGB)
        return resized

    # Load data
    caltech42 = Caltech42(image_processing=process_im)

    # Create the network and train
    network = Network(args,caltech42)
    network.train(caltech42, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(caltech42.test, args):
            print(np.argmax(probs), file=out_file)