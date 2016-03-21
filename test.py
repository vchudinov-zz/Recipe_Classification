import unittest
import theano_mlp
import theano
import theano.tensor as T
import numpy

class mlp_Layer_test(unittest.TestCase):

    def test_init(self):
        in_count = 3
        out_count = 1
        activation = None
        W = [0.5, 0.5, 0.5]
        b = [1 ]
        rng = numpy.random.RandomState(1234)
        layer = theano_mlp.MLP_Layer(rng, in_count, out_count, W , b, activation)
        assert layer.params == [W, b]
        assert layer.activation == activation
        return

    def relu(self, inp):
        return theano.tensor.switch(inp < 0, 0, inp)

    def test_output(self):

        inp = [0., 1., 1.]
        in_count = 3
        out_count = 1
        W = [0.5, 0.5, 0.5]
        b = [0.0]
        rng = numpy.random.RandomState(1234)
        layer = theano_mlp.MLP_Layer(rng,n_in = in_count, n_out = out_count, W  = W, b = b, activation = None)

        activations = [
                        None,
                        theano.tensor.nnet.sigmoid,
                        self.relu # NOT WORKING
                      ]

        for activation in  activations:

            layer.activation = activation

            if activation is None:
                expected = T.dot(inp, W) + b
            else:
                layer.activate(inp)
                expected = activation(T.dot(inp, W) + b)

            assert ( expected.eval() == layer.output )

        return

class neural_network_test(unittest.TestCase):

    def test_init(self):
        return False
