import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import pprint
import numpy as np
from matplotlib.colors import ListedColormap


class Sample:
    def __init__(self, features, label):
        self.features = features
        self.label = label


samples = []

for number in range(2):
    path = f"./{number}/"

    print("Path to dataset files:", path)

    count = 0
    for file in os.listdir(path):
        print(file)
        image = Image.open(path + file)
        raster = image.load()
        value = 0
        for y in range(image.height):
            for x in range(image.width):
                pixel = raster[x, y]
                value += pixel

        value /= image.height*image.width
        print("sum: " + str(value))
        samples.append(Sample([value], number))

        count += 1
        if count >= 2:
            break

pprint.pprint(list(map(lambda x: x.__dict__, samples)))

# Build a perceptron
weights = [random.random()*2-1 for _ in range(len(samples[0].features)+1)]
# weights = [0, 0]
# Show the weights when we start
print(weights)


# Predict the label of a given sample using the weights in the perceptron
def predict_label(sample):
    features = sample.features
    sum = 0
    for i in range(len(features)):
        sum += features[i]*weights[i]
    sum += weights[-1]
    return 0 if sum < 0 else 1

# How fast we update the perceptron
learning_rate = .01

# Now do some training
weight_history = []
for epoch in range(1000):
    for sample in samples:
        predicted_label = predict_label(sample)
        error = sample.label - predicted_label
        if error == 0:
            # We got it right, do nothing
            pass
        else:
            for i in range(len(sample.features)):
                weights[i] += learning_rate*error * sample.features[i]
            weights[-1] += learning_rate * error
    # print(weights)
    weight_history.append(weights.copy())
# print(weight_history)

#Show the weights when we end
print(weights)



# Show a plot of the number and their average whiteness


# Fill up the heat map data 
heatmap_data = np.empty((1, 255)) 

for i in range(255):
    heatmap_data[0, i] = predict_label(Sample([i], None))

# Generate the scatter plot for original image information
x_data = list(map(lambda x: x.features[0], samples))
y_data = [0] * len(samples)

# Generate the scatter plot to show how the perceptron evolved
x_data_2 = list(map(lambda x:x[0], weight_history))
y_data_2 = list(map(lambda y:y[1], weight_history))

# Generate the figure
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16,8))

# Show the heatmap values
heatmap_im = ax.imshow(heatmap_data, cmap=ListedColormap([(1, 0, 0, .5), (0, 1, 0, .5)]))

# Show what the heatmap values mean
plt.colorbar(heatmap_im, ax=ax, label="Predicted Values")

#Add the scatter plot values
ax.scatter(x_data, y_data)

#Label the scatter plot values for the original image data
for sample in samples:
    ax.text(sample.features[0]+.1, 0, str(sample.label))


#Add the scatter plot values for the evolution of the perceptron
ax2.scatter(x_data_2, y_data_2)
ax2.plot(x_data_2, y_data_2)

# Set limits on the perceptron history display
ax2.set_xlim(min(x_data_2), max(x_data_2))
ax2.set_ylim(min(y_data_2), max(y_data_2))


ax.set_xlim(0, 100)
ax.set_xlabel("Average Whiteness")
ax.set_title("Perceptron")
ax.set_aspect(10)
plt.show()
