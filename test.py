import unittest
import theano_mlp
import theano
import theano.tensor as T
import numpy

class hidden_layer_test(unittest.TestCase):

    def test_init(self):
        inp = [0, 0, 0 ]
        in_count = 3
        out_count = 1
        activation = None
        W = [0.5, 0.5, 0.5]
        b = [1, 1, 1]
        rng = numpy.random.RandomState(1234)
        layer = theano_mlp.Hidden_Layer(rng, inp, in_count, out_count, W , b, activation)
        assert layer.input == inp
        assert layer.params == [W, b]

        return

    def test_output(self):
        inp = [0, 1, 1 ]
        in_count = 3
        out_count = 1
        W = [0.5, 0.5, 0.5]
        b = [ 0 ]
        rng = numpy.random.RandomState(1234)
        # Linear
        activation = None
        layer = theano_mlp.Hidden_Layer(rng, input = inp, n_in = in_count, n_out = out_count, W  = W, b = b, activation = activation)
        expected = T.dot(inp, W) + b

        print 'expected: ', expected.eval()
        print 'actual: ', layer.output.eval()

        assert ( expected.eval() == layer.output.eval() ).all()
        # sigmoid
        activation = theano.tensor.nnet.sigmoid

        layer = theano_mlp.Hidden_Layer(rng, input = inp, n_in = in_count, n_out = out_count, W = W , b = b, activation = activation)
        W *= 4

        expected = activation(inp, W) + b

        print 'expected: ', expected.eval()
        print 'actual: ', layer.output.eval()
        assert ( expected.eval() == layer.output.eval() ).all()





        # Sigmoid
        activation = T.sigmoid


        return





    def relu_test(self):
        return False

class output_layer_test(unittest.TestCase):

    def test_init(self):
        return False

class neural_network_test(unittest.TestCase):

    def test_init(self):
        return False
