#taken from http://deeplearning.net/tutorial/mlp.html
class Hidden_Layer(object):
    def __init__(self, rng, input, n_in, n_out, W = None, b = None,
    activation = T.tanh):
    self.input = input

    if W is None:
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
        self.logRegressionLayer = LogisticRegression(
        input = self.hiddenLayer.output,
        n_in = n_hidden,
        n_out = n_out
        )
        self.add_hidden_layer(count, rng, input, n_in, n_out, activation):
            self.hiddenLayers.append(HiddenLayer(
            rng = rng,
            input = input
            n_in = n_in
            n_out = n_out
            activation = activation))
        # L1 and L2 regularization
        self.L1 = (abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negtive_log_likelihood = (
           self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input

def epoch(inputs):
    for i in inputs:
        avg_cost = train_model(i)
def train()
   x = T.ivector('x') #input
   y = T.ivector('y)' #output

   netowrk = Neural_Network(
    rng = rng,
    input = x
    n_in = 9999 #change to len vocab
    n_hidden = n_hidden
    n_out = 10 # change to classes

   )

   cost = network.negative_log_likelihood(y) + L1_reg * network.L1 + L2_reg * network.L2_sqr
