import neuralnetwork as nn
import numpy as np
import matplotlib.pyplot as plt

with np.load('Mnist-data-numpy-format/mnist.npz') as data: 
    training_images = data['training_images']
    training_labels = data['training_labels']

plt.imshow(training_images[0].reshape(28,28), cmap = 'gray')
plt.show

#print(data.files) #print all data in this file

print(training_images.shape)
print(training_labels.shape)

layer_sizes = (3,5,10)
x = np.ones((layer_sizes[0],1))

net = nn.NeuralNetwork(layer_sizes)
prediction = net.predict(x)

print(prediction)

# 28x28 images = 784 pixels - ranging from 0 to 255 (gray scale only)