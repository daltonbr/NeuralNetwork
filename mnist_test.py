import numpy as np

with np.load('Mnist-data-numpy-format/mnist.npz') as data: 
	training_images = data['training_images']

print(training_images)
