#!/usr/bin/python3

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from sklearn import metrics

from BayesNetworks import Standard_NN, Gaussian_BBB_NN, Reparameterised_Gaussian_BBB_NN

class Gaussian_BBB_NN(Gaussian_BBB_NN):
    def model(self, X, samples=1,training=True):
        """
        Returns logit outputs of the model, with shape [sample, batch, logit]
        """

        # Transforms inputs from [batch, input] into [sample, batch, input]
        X = tf.tile(tf.expand_dims(X, 0), [samples, 1, 1])        

        z = self.fully_connected_layer(X, samples, size=(28*28, 256), name="layer_one")
        h = tf.nn.relu(z)
        z = self.fully_connected_layer(h, samples, size=(256,256), name="layer_two")
        h = tf.nn.relu(z)
        z = self.fully_connected_layer(h, samples, size=(256, 10), name="layer_three")
        logits = z
        # logits should still be of shape [sample, batch, logit]
        return logits

def continual_learning():
    (X, Y), (X_test, Y_test) = mnist.load_data()

    Y_test = tf.one_hot(Y_test, 10, 1.0, 0.0)
    X_test = tf.reshape(X_test, (-1,28*28))/156.
    
    mask = np.zeros_like(Y_test)
    for i in range(5):
        mask[:,i*2] = np.logical_or(Y_test[:,i*2],Y_test[:,i*2+1])
        mask[:,i*2+1] = mask[:,i*2]

    #  model = Standard_NN() 
    model = Gaussian_BBB_NN(num_training_samples=1, num_pred_samples=5,KL_weight=1) 
    #  model = Reparameterised_Gaussian_BBB_NN(num_training_samples=1, num_pred_samples=5,KL_weight=1) 
    accuracies_history = []
    for i in range(5):
        i = i%5

        ints = [i*2,i*2+1]

        print(f"\n\nTraining on data with label: {ints[0]} and {ints[1]}")
        _Y = Y[np.logical_or(Y==ints[0], Y==ints[1])]
        _X = X[np.logical_or(Y==ints[0], Y==ints[1])]
        _Y = tf.one_hot(_Y, 10, 1.0, 0.0)
        _X = tf.reshape(_X, (-1,28*28))/156.

        for loss in model.train(_X,_Y,num_epochs=2,batch_size=64,step_size=0.001):
            print(f"Loss: {loss}")

        testPredictions = model.predict(X_test)
        y_true = np.argmax(Y_test.numpy(),axis=-1)
        y_pred = np.argmax(testPredictions*mask,axis=-1)
        accuracies = []
        for j in range(5):
            indicies = np.logical_or(y_true==j*2,y_true==j*2+1)
            accuracy = np.sum(y_pred[indicies] == y_true[indicies])/len(y_true[indicies])
            accuracies.append(accuracy)

        print(accuracies)
        accuracies_history.append(accuracies)

        posterior = model.get_posterior()
        model.set_prior(posterior)

    print(np.array(accuracies_history)) 

    #  y_pred = np.argmax(testPredictions,axis=-1)
    #  accuracy = np.sum(y_pred == y_true)/len(Y_test.numpy())
    #  print(f"Test Accuracy: {accuracy*100:.2F}%")
    #  print(metrics.confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    continual_learning()
