import numpy as np
import json as js
import matplotlib.pyplot as plt
with open("train.json") as ts:
        training_set = js.load(ts)

labels = []
ingredients = []
x = []
y = []
explore = {}

for entry in training_set:
    label = entry['cuisine']
    if label not in explore:
        explore[label] = 0
    explore[label] += 1

    for ingredient in entry['ingredients']:
        if ingredient not in ingredients:
            ingredients.append(ingredient)

    x.append(entry['ingredients'])
    y.append(label)

labels = list(explore.keys())
# Create and visualise categories.
print ("Category Count: %d" %(len(labels)))
print ("Category \t Size")
for label in labels:
    print(label + " \t " + str(explore[label]))

for lab in range(len(y)):
    label = [0 for x in range(len(explore))]
    label[labels.index(y[lab])] = 1
    y[lab] = label

for i in range(10):
    print( x[i], " ", str(y[i] ) )

#fig, ax = plt.subplots()
#rects = ax.bar( [ x for x in range(1, len(explore) +1 ) ], height=list(explore.values()), width=0.35, color='blue')
#ax.set_xticks([ x for x in range(1, len(explore) +1 ) ])
#ax.set_xticklabels(list(explore.keys()) )

#plt.savefig('food.png')

# Now lets turn everything into vectors.
