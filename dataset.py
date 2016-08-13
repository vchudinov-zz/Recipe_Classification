import json as js
import numpy as np
'''
Class + methods for holding a dataset in the form of [[x], [y]].
'''
class dataset():

    def __init__(self, dtype = 'train'):
        self.data = [[], []] # x and y
        self.data_size =0
        self.type = dtype # train, test, validate
        self.labels = [] # label names
        self.vocabulary = [] # for language tasks
        self.category_sizes = 0

        return

# the name assigned to labels. i.e. with the cooking dataset it is cuisine
    def load_from_json(self, filename, y_name, x_name):
        x, y = [], []
        datadict = {}
        with open(filename) as ds:
            raw_data = js.load(ds)
        for entry in raw_data:
            label = entry[y_name]
            if label not in self.labels:
                self.labels.append(label)
            for ingredient in entry[x_name]:
                if ingredient not in self.vocabulary:
                    self.vocabulary.append(ingredient)

            x.append(entry[x_name])
            y.append(label)
            self.hot_vectorize_data(x,y)
        return

    def hot_vectorize_data(self, x, y):

        for i in range(len(y)):
            # turn label into vector
            label = [0.0 for x in range(len(self.labels))]
            label[self.labels.index(y[i])] = 1.0
            # turn data in one hot vector
            X_i = [0.0 for x in range(len(self.vocabulary))]
            for ingredient in x[i]:
                X_i[self.vocabulary.index(ingredient)] = 1.0

            self.data[0].append(X_i)
            self.data[1].append(label)
        return

# returns next sequential minibatch
    def next_minibatch(self, batch_size):
        self.counter = 0;
        while counter*batch_size+batch_size > len(self.data[0]):

            start_index = counter*batch_size
            end_index = start_index + batch_size

            yield [data[0][start_index:end_index], data[1][start_index:end_index]]
            counter+=1

# Returns a minibatch from uniform random selected examples from the dataset
    def next_random_minibatch(self, batch_size):
        indexes = np.random.randint(low=0, high=self.data_size,
                                size=self.batch_size, dtype="int")

        return [[self.data[0][x] for x in indexes], [self.data[1][y] for y in indexes]]
