#!/usr/bin/python3

import numpy as np
import tensorflow as tf

from BayesNetworks import Gaussian_BBB_NN

class Model_Selecting_BBB_NN(Gaussian_BBB_NN):

    def model_selection_fully_connected_layer(self, x, samples, size, name):
        #double the output size
        size = (*size[:-1],size[-1]*2)
        output = self.fully_connected_layer(x, samples, size, name)
        output, dropout_logits = tf.split(output, 2, axis=-1)
        dropout = tf.less(tf.sigmoid(dropout_logits*10-2),tf.random_uniform(tf.shape(output)))
        #  print(np.sum(dropout.numpy())/np.prod(tf.shape(dropout)))
        return tf.cast(dropout,tf.float32)*output

    def model(self, X, samples=1,training=True):
        """
        Returns logit outputs of the model, with shape [sample, batch, logit]
        """

        # Transforms inputs from [batch, input] into [sample, batch, input]
        X = tf.tile(tf.expand_dims(X, 0), [samples, 1, 1])        

        z = self.model_selection_fully_connected_layer(X, samples, size=(28*28, 128), name="layer_one")
        h = tf.nn.relu(z)
        z = self.model_selection_fully_connected_layer(h, samples, size=(128, 64), name="layer_two")
        h = tf.nn.relu(z)
        z = self.model_selection_fully_connected_layer(h, samples, size=(64, 10), name="layer_three")
        logits = z
        # logits should still be of shape [sample, batch, logit]
        return logits
