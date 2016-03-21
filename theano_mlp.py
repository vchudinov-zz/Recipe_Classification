# Code is at this point taken from http://deeplearning.net/tutorial/mlp.html
# Comments are mine - trying to understand how the code works, so I could later
# adapt it to my needs.

import sys
import theano.tensor as T
import theano
import numpy

class MLP_Layer(object):

    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation=T.tanh, is_output = False):

        self.input = input

        if W is None:
            # Create a matrix for weights
            W_values = numpy.asarray(
                rng.uniform(
                    low=numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.activation = activation
        self.params = [self.W, self.b]
        self.output = None
        # Predicted class
        self.y_pred = None

    def activate(self, input):

        self.input = input

        lin_output = T.dot(self.input, self.W) + self.b

        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

        self.probability_distribution(input)

        return

    def probability_distribution(self, input):
        if self.output is None:
            self.output = self.activate(input)
        self.p_y_given_x = T.nnet.softmax(self.output)

    def predict(self):
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x), [T.arange(y.shape[0], y)])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same share as y.pred',
                ('y', y.type, 'ypred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class Neural_Network(object):

    def __init__(self, n_in, n_hidden, n_out, rng):
        # One hidden layer.
        if rng is None:
            rng = rng.numpy.random.RandomState(1234)
        self.hiddenLayer = MLP_Layer(
            rng=rng,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.sigmoid
        )

        # This is output layer
        self.output_layer = MLP_Layer(
            rng = rng
            n_in=n_hidden,
            n_out=n_out,
            activation = T.tanh
        )
        #layers = []

        # L1 and L2 regularization. Figure how to add for multiple layers
        self.L1 = (abs(self.hiddenLayer.W).sum()
                   + abs(self.logRegressionLayer.W).sum()
                   )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )


        # List of parameters
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # Not
        self.index = T.lscalar  # index of minibatch
        self.x = T.matrix('x')  # Used for input data.
        self.y = T.ivector('y')  # Used for labels.


# shared variables for stuff.
# do I need neglog and errors for hidden?
    def activate(self, x, y):
        self.input = input

        hidden_layer.activate(self.input)
        output_layer.activate(hidden_layer.output)

        self.negtive_log_likelihood = (
            self.output_layer.negative_log_likelihood(y)
        )

        self.errors = self.output_layer.errors

        return self.negative_log_likelihood

    def dropout(self):
        raise NotImplementedError()

    def add_hidden_layer(self, rng, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        raise NotImplementedError()

    def update_model(self, x, y):

        cost = (self.activate(x, y)
                + L1_reg * self.L1
                + L2_reg * self.L2_sqr
                )

        gradients = [T.grad(cost, param) for param in self.params]

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gradients)
        ]

        new_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # alternative. not as optimal on big datasets.
        # updates just updates a given variable according to a given expression i.e. (var1, var1 + 1)
        # new_model = theano.function(
        # inputs = [train_set_x[index*batch_size : (index +1)*batch_size],
        # train_set_y[index*batch_size : (index +1)*batch_size]]
        #outputs = cost
        #updates = updates
        #)

        return new_model(x)

    def test_model(self, index, y):
        test_model = theano.function(
            inputs=[index],
            outputs=self.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
        return test_model

    def validate_model(self, x):
        raise NotImplementedError()

    def train_model(self, n_epochs):
        patience = 10000
        patience_increase = 2
        improvement_threshld = 0.995
        validation_frequency = min(n_train_batches, patience // 2)

        best_loss = numpy.inf
        best_iter = 0
        test_score = 0
        start_time = timeit.default_timer()
        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = update_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = [
                        validate_model(i) for i in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                if this_validation_loss < best_validation_loss * improvement_threshld:
                    patience = max(patience, iter * patience_increase)
                best_validation_loss = this_validation_loss
                best_iter = iter

                test_losses = [test_model(i) for i in range(n_test_batches)]
                test_score = numpy.mean(test_losses)

            if patience <= iter:
                done_looping = True
                break

    def save_model(self, save_file):
        raise NotImplementedError()

    def load_model(self, load_file):
        raise NotImplementedError()

    def pring_model(self):
        raise NotImplementedError()

    def visualize_model(self):
        raise NotImplementedError()
