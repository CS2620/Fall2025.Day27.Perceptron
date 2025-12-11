from PIL import Image
import os

class Sample:
    def __init__(self, features, label):
        self.features = features
        self.label = label
        
class Perceptron:
    def __init__(self, weights):
        self.weights = weights
    
    def get_main_weights(self):
        return self.weights[:-1]
    
    def get_bias(self):
        return self.weights[-1]
 
samples = [] 
 
for number in [0, 1]:
    path = f"./{number}/"
    files = os.listdir(path)
    for i in range(len(files)):
        file_path = path + files[i]
        image = Image.open(file_path)
        raster = image.load()
        sum = 0
        for y in range(image.height):
            for x in range(image.width):
                pixel = raster[x,y]  
                sum += pixel
        average = sum / (image.height*image.width)
        samples.append(Sample([average], number))

print(samples)
print(samples[0].features)
print(samples[1].features)

perceptron = Perceptron([0,0])

def guess_label(s, p):
    features = s.features
    main_weights = p.get_main_weights()
    sum = 0
    for i in range(len(features)):
        sum += features[i] * main_weights[i]
    sum += p.get_bias()
    #Time for a squashing function
    return 0 if sum < 0 else 1

print(guess_label(samples[0], perceptron))
print(guess_label(samples[1], perceptron))

# Perceptron Update Rule

epochs = 10000
training_rate = .1

for epoch in range(epochs):
    for sample in samples[:-1]:
        predicted_label = guess_label(sample, perceptron)
        if predicted_label != sample.label:
            features = sample.features
            main_weights = perceptron.get_main_weights()
            for i in range(len(features)):
                perceptron.weights[i] += (sample.label - predicted_label)*training_rate*features[i]
            perceptron.weights[-1] += (sample.label - predicted_label)*training_rate
    print(perceptron.weights)
    
print(guess_label(samples[0], perceptron))
print(guess_label(samples[1], perceptron))
print(guess_label(samples[-1], perceptron))
