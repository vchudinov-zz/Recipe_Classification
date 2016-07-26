import numpy as np
import json as js
import matplotlib.pyplot as plt
from itertools import islice, chain
with open("train.json") as ts:
        training_set = js.load(ts)
import tensorflow as tf
labels = []
vocabulary = []
x = []
y = []
explore = {}

for entry in training_set:
    label = entry['cuisine']
    if label not in explore:
        explore[label] = 0
    explore[label] += 1

    for ingredient in entry['ingredients']:
        if ingredient not in vocabulary:
            vocabulary.append(ingredient)

    x.append(entry['ingredients'])
    y.append(label)

labels = list(explore.keys())
# Create and visualise categories.
print ("Category Count: %d" %(len(labels)))
print ("Category \t Size")
for label in labels:
    print(label + " \t " + str(explore[label]))

print ("Vocabulary size: %d" %(len(vocabulary)))
#labels
data = [[],[]]

for i in range(len(y)):
    # turn label into vector
    label = [0.0 for x in range(len(explore))]
    label[labels.index(y[i])] = 1.0
    # turn data in one hot vector
    X_i = [0.0 for x in range(len(vocabulary))]
    for ingredient in x[i]:
        X_i[vocabulary.index(ingredient)] = 1.0

    data[0].append(X_i)
    data[1].append(label)
# Returns a random minibatch derived from dataset iterable
class batch_iterator:

    def __init__(self, iterable, batch_size):

        self.data_size = len(iterable[0])
        self.iterable = iterable
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self):
        indexes = np.random.randint(low=0, high=self.data_size, size=self.batch_size, dtype="int")

        return [[data[0][x] for x in indexes], [data[1][y] for y in indexes]]

def generator_batch(data, batch_size):
    self.counter = 0;
    while counter*batch_size+batch_size > len(data[0]):

        start_index = counter*batch_size
        end_index = start_index + batch_size

        yield [data[0][start_index:end_index], data[1][start_index:end_index]]

input_size = len(vocabulary)
categories =  len(labels)
#returns a variable of a given shape. Used for weights
#gen = batch_iterator(data, 1000)
#for i in range(60):
#    batch = gen.next()



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def relu_activation(W, x):
    return tf.relu(tf.matmul(x,W) + b)

def softmax(W, x):
    return tf.nn.softmax()

print ("Setting up Vairables")
#W = tf.Variable(tf.zeros())
#b = tf.Variable(tf.zeros([categories]))
tf_x = tf.placeholder(tf.float32, shape=[None, input_size])
y_expected = tf.placeholder(tf.float32, shape=[None, categories])

#first hidden
W_h1 = weight_variable([input_size, 1000])
b_h1 = bias_variable([1000])

y_h1 = tf.nn.relu(tf.matmul(tf_x,W_h1) + b_h1)

# second hidden layer
W_h2 = weight_variable([1000,500])
b_h2 = bias_variable([500])

y_h2 = tf.nn.relu(tf.matmul(y_h1,W_h2) + b_h2)

#Last Hidden layer
#W_h3 = weight_variable([1000, categories])
#b_h3 = bias_variable([categories])
W_o = weight_variable([500, categories])
b_o = bias_variable([categories])

y_predicted = tf.nn.softmax(tf.matmul(y_h2, W_o) + b_o)


print("Variables Set. Setting up calculations")

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_expected * tf.log(y_predicted), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_expected,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(" Begin Training")
sess = tf.Session()
sess.run(tf.initialize_all_variables())
gen = batch_iterator(data, 20)

for i in range(10000):
    batch = gen.next()
    if i%50 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={tf_x: batch[0], y_expected: batch[1]})
        print("Epoch %d, training accuracy %g" %(i, train_accuracy))
    train_step.run(session = sess, feed_dict={tf_x: batch[0], y_expected: batch[1]})
final_accuracy = accuracy.eval(session=sess, feed_dict={tf_x: data[0], y_expected: data[1]})
print("Final accuracy: %g" %(final_accuracy))
