import numpy as np
import theano
import theano.tensor as T
import json as js

###############
# TO DOs:
# 1. Train NN
# 2. Get detailed data about each class
# 3. Figure out all the interesting things
# 4. Add visualizations.
##############
training_set = []
with open("train.json") as ts:
        training_set = js.load(ts)

vocabulary = []
labels = []

for entry in training_set:

    if entry['cuisine'] not in labels:
        labels.append(entry['cuisine'])
    for ingredient in entry['ingredients']:
        if ingredient not in vocabulary:
            vocabulary.append(ingredient)
#create a sparse matrix of the data
doc_matrix = np.zeros((len(training_set), len(vocabulary)), dtype = np.int)
label_matrix = np.zeros( (len(training_set), len(labels)), dtype = np.int)

for entry in range(len(training_set)):
    label_matrix[entry][labels.index(training_set[entry]['cuisine'])] = 1
    for ingredient in training_set[entry]['ingredients']:
        doc_matrix[entry][vocabulary.index(ingredient)] = 1
# най-дървения начин за принтиране
print "Total number of documents " + str(len(doc_matrix))
print "Total number of labels " + str(len(labels))
print "Vocabulary size " + str(len(vocabulary))

train = doc_matrix[:30000]
train_labels = label_matrix[:30000]
validate = doc_matrix[30001:]
def init_weights(n_hidden, n_output):
    #create 0.0
    weigths = numpy.zeros((n_hidden, n_out), dtype = theano.config.floatX) #?????
    bias = numpy.zeros((n_out), dtype = theano.config.floatX)
    return (
    theano.shared( name = 'W', borrow = True, value = weight) # ??
    theano.shared( name = 'b', borrow = True, value = bias)  #???
    )

def init_hidden_weights(n_in, n_out):
    rng = numpy.random.RandomState(1234) # ???
    weights = numpy.asarray(
        rng.uniform(
            low = numpy.sqrt(6. / n_int + n_out),
            high = numpy.sqrt(6. / n_int + n_out),
            size = (n_in,n_out)
        ),
        dtype = theano.config.floatX
    )
    bias = numpy.zero((n_out), dtype = theano.config.floatX)
    return (
      theano.shared(value = weights, name = 'W', borrow = True)
      theano.shared(value = bias, name = 'b', borrow = True)
      )

def feed_forward(activation, weights, bias, input_):
    return activation(T.dot(input_, weights) + bias)

    
    raise Exception ("Not Implemented")
def epoch():
    raise Exception ("Not Implemented")
def train(training_set):
    raise Exception ("Not Implemented")
def evaluate(input):
    raise Exception ("Not Implemented")
def test(test_set):
    raise Exception ("Not Implemented")
