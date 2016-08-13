
import tensorflow as tf
import dataset

data = dataset.dataset()
data.load_from_json("train.json", 'cuisine', 'ingredients')

input_size = len(data.vocabulary)
categories =  len(data.labels)

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

#placeholders
tf_x = tf.placeholder(tf.float32, shape=[None, input_size])
y_expected = tf.placeholder(tf.float32, shape=[None, categories])

#first hidden layer params.
#size is [lower_layer_size, upper_layer_size]
W_h1 = weight_variable([input_size, 1000])
b_h1 = bias_variable([1000])
# the output of the layer. relu(x*W^T + b)
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

#Training
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_expected * tf.log(y_predicted),
                                reduction_indices=[1]))
# step repeats.
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_expected,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(" Begin Training")
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(10000):
    batch = data.next_random_minibatch(5)
    if i%50 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={tf_x: batch[0], y_expected: batch[1]})
        print("Epoch %d, training accuracy %g" %(i, train_accuracy))
    train_step.run(session = sess, feed_dict={tf_x: batch[0], y_expected: batch[1]})
final_accuracy = accuracy.eval(session=sess, feed_dict={tf_x: data[0], y_expected: data[1]})
print("Final accuracy: %g" %(final_accuracy))
