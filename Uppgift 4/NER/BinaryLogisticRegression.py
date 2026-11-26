from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418 at KTH.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.1  # The learning rate.
    CONVERGENCE_MARGIN = 0.00001  # The convergence criterion.
    MAX_ITERATIONS = 10 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)
    MINIBATCH_MAX_EPOCHS = 1000 # Maximal number of passes through the data in minibatch gradient descent.

    PLOT = True           # Plot the loss during training? Set to False when the code works (plotting is slow).
    PLOT_EVERY = 10       # Plot every n:th iteration (batch/minibatch). Stochastic plots every 100*n:th step.

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def loss(self, x, y):
        """
        Computes the loss function given the input features x and labels y
        
        :param      x:    The input features
        :param      y:    The correct labels
        """
        # Cross-entropy: -(1/m) * sum( y*ln(p) + (1-y)*ln(1-p) )
        p = self.sigmoid(np.dot(x, self.theta))
        eps = 1e-12                        # protects against ln(0)
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # P(1|d) = sigma(theta^T x(d)), and P(0|d) = 1 - P(1|d)
        p1 = self.sigmoid(np.dot(self.theta, self.x[datapoint]))
        return p1 if label == 1 else 1 - p1


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """
        # gradient[k] = (1/m) * sum_i x_k(i) * (sigma(theta^T x(i)) - y(i))
        error = self.sigmoid(np.dot(self.x, self.theta)) - self.y
        self.gradient = np.dot(self.x.T, error) / self.DATAPOINTS
        return self.gradient


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        'minibatch' is an array with the indices of the datapoints in the batch.
        """
        xb = self.x[minibatch]
        error = self.sigmoid(np.dot(xb, self.theta)) - self.y[minibatch]
        self.gradient = np.dot(xb.T, error) / len(minibatch)
        return self.gradient


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        'datapoint' is the index of the datapoint.
        """
        xi = self.x[datapoint]
        self.gradient = xi * (self.sigmoid(np.dot(self.theta, xi)) - self.y[datapoint])
        return self.gradient


    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        if self.PLOT:
            self.init_plot(1)      # one curve: the loss

        step = 0
        for epoch in range(self.MAX_ITERATIONS):
            # One epoch = DATAPOINTS steps, each on one randomly chosen datapoint.
            for _ in range(self.DATAPOINTS):
                i = random.randrange(self.DATAPOINTS)
                self.theta = self.theta - self.LEARNING_RATE * self.compute_gradient(i)
                step += 1
                if self.PLOT and step % (100 * self.PLOT_EVERY) == 0:
                    self.update_plot(self.loss(self.x, self.y))

            # Check convergence on the gradient of the whole dataset after each epoch.
            if np.sum(self.compute_gradient_for_all() ** 2) < self.CONVERGENCE_MARGIN:
                break
        print('Stochastic gradient descent: {} epochs, {} steps'.format(epoch + 1, step))


    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        if self.PLOT:
            self.init_plot(1)

        indices = np.arange(self.DATAPOINTS)
        step = 0
        for epoch in range(self.MINIBATCH_MAX_EPOCHS):
            np.random.shuffle(indices)       # new random batches every epoch
            for start in range(0, self.DATAPOINTS, self.MINIBATCH_SIZE):
                minibatch = indices[start:start + self.MINIBATCH_SIZE]
                self.theta = self.theta - self.LEARNING_RATE * self.compute_gradient_minibatch(minibatch)
                step += 1
                if self.PLOT and step % self.PLOT_EVERY == 0:
                    self.update_plot(self.loss(self.x, self.y))

            if np.sum(self.compute_gradient_for_all() ** 2) < self.CONVERGENCE_MARGIN:
                break
        print('Minibatch gradient descent: {} epochs, {} steps'.format(epoch + 1, step))


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        if self.PLOT:
            self.init_plot(1)

        iteration = 0
        while True:
            gradient = self.compute_gradient_for_all()
            self.theta = self.theta - self.LEARNING_RATE * gradient
            iteration += 1
            if self.PLOT and iteration % self.PLOT_EVERY == 0:
                self.update_plot(self.loss(self.x, self.y))
            # Converged when the sum of the squared partial derivatives is small.
            if np.sum(gradient ** 2) < self.CONVERGENCE_MARGIN:
                break
        print('Batch gradient descent: {} iterations'.format(iteration))


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))

        # confusion[predicted][real]. Class 1 = 'name', class 0 = 'no name'.
        tp, fp = confusion[1][1], confusion[1][0]
        fn, tn = confusion[0][1], confusion[0][0]

        def ratio(a, b):
            return a / b if b > 0 else 0.0     # avoid division by zero

        print()
        print('Accuracy:             {:.4f}'.format(ratio(tp + tn, tp + tn + fp + fn)))
        print('Precision (name):     {:.4f}'.format(ratio(tp, tp + fp)))
        print('Recall    (name):     {:.4f}'.format(ratio(tp, tp + fn)))
        print('Precision (no name):  {:.4f}'.format(ratio(tn, tn + fn)))
        print('Recall    (no name):  {:.4f}'.format(ratio(tn, tn + fp)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, 1+max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()