# Code is at this point taken from http://deeplearning.net/tutorial/mlp.html
# Comments are mine - trying to understand how the code works, so I could later
# adapt it to my needs.

import sys
'''
Each hidden layer has a matrix of Weight W, with shape (n_in, n_out)
Each layer also has a bias vector b, with n_out values.
Output layers isnt much different, but it starts with zero weights apparently.
(TODO: make Output Layer and Hidden Layer be children of a Layer class or something)
'''

class Hidden_Layer(object):

    def __init__(self, rng, input, n_in, n_out, W = None, b = None,
    activation = self.relu):
    self.input = input

    if W is None:
        # Create a matrix for weights
        W_values = numpy.asarray(
            rng.uniform(
                low = numpy.sqrt(6. / (n_in + n_out))
                hight =  numpy.sqrt(6. / (n_in + n_out))
                size (n_in, n_out)
            ),
            dtype = theano.config.floatX
        )
        if activation = theano.tensor.nnet.sigmoid:
            W_values *= 4

        W = theano.shared(value = W_values, name = 'W', borrow = True)

    if b is None:
        b_values = numpy.zeros((n_out,), dtype = theano.config.floatX)
        b = theano.shared(value = b_values, name = 'b', borrow = True)
    self.W = W
    self.b = b

    lin_output = T.dot(input, self.W) + self.b
    self.output = (
        lin_output if activation is None
        else activation(lin_output)
    )

    self.params = [self.W, self.b]

    def relu(self, x):
        return theano.tensor.switch(x<0, 0, x)

# This is the logistic Output Layer
# I think i should be able to use this as basis for linear regression?
class Logistic_Layer():
    def __init__(self, input,activation, n_in, n_out):

        #Weight matrix
        self.W = theano.shared(
            value = numpy.zeros(
                (n_in,n_out),
                dtype = theano.config.floatX
                ),
            name = 'W',
            borrow = True
            )
        #bias vector
        self.b = theano.shared(
            value = numpy.zeros( (n_out,),
            dtype = theano.config.floatX ),
            name = 'b',
            borrow = True
        )

        # Gives a vector of probabilities - the probability distribution of Y|x
        p_y_given_x  = T.nnet.softmax(self.activate(activation, input))

        # Predicted class
        self.y_pred = self.predict

        self.params = [self.W, self.b]

        self.input = input

    def predict():
        return T.argmax(self.p_y_given_x, axis=1)

    def activate(activation, x):
        return activation(T.dot(x, self.W) + self.b)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x), [T.arange(y.shape[0],y)])

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
    
    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.hiddenLayer = HiddenLayer(
        rng = rng,
        input = input
        n_in = n_in
        n_out = n_hidden
        activation = T.sigmoid
        )

        # This is output layer
        self.output_layer = Logistic_Layer(
        input = self.hiddenLayer.output,
        n_in = n_hidden,
        n_out = n_out
        )

        # L1 and L2 regularization. Figure how to add for multiple layers
        self.L1 = (abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )


        self.negtive_log_likelihood = (
           self.output_layer.negative_log_likelihood
        )

        self.errors = self.output_layer.errors

        # List of parameters
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input

#shared variables for stuff.
    def compile_model(self): # Thanks https://gist.github.com/honnibal/6a9e5ef2921c0214eeeb
        index = T.lscalar #index of minibatch
        x = T.matrix('x') #input data.
        y = T.ivector('y') #labels.
        rng.numpy.random.RandomState(1234)


    def update_model(self):
        cost = ( self.negative_log_likelihood(y)
            + L1_reg * self.L1
            + L2_reg * self.L2_sqr
        )

        gradients = [ T.grad(cost, param) for param in self.params ]

        updates = [
        (param, param - learning_rate*gparam)
        for param, gparam in zip(self.params, gradients)
        ]

        train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_set_x[index*batch_size : (index +1)*batch_size],
            y: train_set_y[index*batch_size : (index +1)*batch_size]
            }
        )

# alternative. not as optimal on big datasets.
#updates just updates a given variable according to a given expression i.e. (var1, var1 + 1)
#train_model = theano.function(
#inputs = [train_set_x[index*batch_size : (index +1)*batch_size],
#train_set_y[index*batch_size : (index +1)*batch_size]]
#outputs = cost
#updates = updates
#)
    def test_model(self, index, y):
        test_model = theano.function(
            inputs = [index],
            outputs = self.errors(y),
            givens = {
                x: test_set_x[index*batch_size:(index+1)*batch_size],
                y: test_set_y[index*batch_size:(index+1)*batch_size]
            }
        )

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
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch -1) * n_train_batches + minibatch_index
                if (iter +1 ) % validation_frequency == 0:
                    validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                if this_validation_loss < best_validation_loss * improvement_threshld:
                    patience = max(patience, iter*patience_increase)
                best_validation_loss = this_validation_loss
                best_iter = iter

                test_losses = [test_model(i) for i in range(n_test_batches)]
                test_score = numpy.mean(test_losses)
            if patience <= iter:
                done_looping = True
                break

# parameter update as per backprop
network = Neural_Netwokr(
rng =rng
input = x
n_in = 28*28, #this one is for MINST
n_hidden = n_hidden,
n_out = 10
)

train_model = theano.function(
    inputs = [index],
    outputs = cost,
    updates = updates,
    givens = {
    x: train_set_x[index * batch_size: (index +1) *batch_size],
    y: train_set_y[index * batch_size: (index +1) *batch_size]
    }
)
# training
