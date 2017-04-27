import numpy as np
import random
import math
import os.path

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

class AutoEncoder(object):

	def __init__(self, sizes):
		print("Initializing AutoEncoder with sizes " + str(sizes))
		self.sizes = sizes
		self.num_layers = len(sizes)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def GetParameters(self):	
		return self.weights
		
	def SaveParameters(self, path):
		print("Saving AutoEncoder Parameters")
		with open(path, "w+") as file:
			for layer in self.weights:
				data = np.reshape(layer, (layer.shape[0]*layer.shape[1], 1))
				for x in data:
					file.write(str(x[0]) + ",")
				file.write("\n")
			for layer in self.biases:
				data = np.reshape(layer, (layer.shape[0]*layer.shape[1], 1))
				for x in data:
					file.write(str(x[0]) + ",")
				file.write("\n")
				
			file.close
	
	def LoadParameters(self, path):
		print("Loading AutoEncoder Parameters")
		
		if(not os.path.exists(path)):
			print("File " + path + " doesn't exist")
			return False
		
		temp = []
		with open(path, 'r') as file:
			data = file.readlines()
			for line in data:
				for point in [x.strip() for x in line.split(',')]:
					if(len(point) > 0): temp.append(float(point))
			file.close
		
		expectedSize = (self.sizes[0]*self.sizes[1])+(self.sizes[1]*self.sizes[2])+(self.sizes[1])+(self.sizes[2])
		if(len(temp) != expectedSize):
			print("Error Loading Parameters: expected size is " + str(expectedSize) + " but " + str(len(temp)) + " parameters were found in file")
			return False

		counter = 0
		for layer in self.weights:
			for r in range(0, int(layer.shape[0])):
				for c in range(0, int(layer.shape[1])):
					layer[r][c] = temp[counter]
					counter = counter + 1
		for layer in self.biases:
			for r in range(0, int(layer.shape[0])):
				for c in range(0, int(layer.shape[1])):
					layer[r][c] = temp[counter]
					counter = counter + 1
		print(str(counter) + " parameters effectively loaded")
		return True
	
	def FeedForward(self, input):
		for b, w in zip(self.biases, self.weights):
			input = sigmoid(np.dot(w, input) + b) #was + b
		return input
	
	def CalculateCost(self, input, output):
		rate = 0.0
		for a, b in zip(np.array(input), np.array(output)):
			rate = rate + abs(a - b)
		return rate
	
	def Train(self, trainingData, learningRate, lmbda, epochs, batchSize, path=None):
		print("Training AutoEncoder")
		n = len(trainingData)
		
		cost = 0.0
		for epoch in xrange(epochs):
			cost = 0.0
			random.shuffle(trainingData)
			miniBatches = [trainingData[k:k+batchSize] for k in xrange(0, n, batchSize)]
			for miniBatch in miniBatches:
				cost = cost + self.CalculateCost(miniBatch[0][0], self.FeedForward(miniBatch[0][0]))
				self.UpdateMiniBatches(miniBatch, learningRate, lmbda, len(trainingData))			
			print ("epoch {0} completed. Learning Rate: {1}, Cost: {2:8f}".format(epoch, learningRate, float(cost/batchSize)))
			if(path != None and epoch % 1000 == 0):
				self.SaveParameters(path)
		
		print("Training AutoEncoder Completed")
		if(path != None):
			self.SaveParameters(path)		
		return cost
			
	def UpdateMiniBatches(self, miniBatch, learningRate, lmbda, n=0):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		for x, y in miniBatch:
			delta_nabla_b, delta_nabla_w = self.BackPropagate(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		
		if(n == 0 or lmbda == 0):
			self.weights = [w-(learningRate/len(miniBatch))*nw for w, nw in zip(self.weights, nabla_w)]
		else:
			self.weight = [(1-learningRate*(lmbda/n))*w-(learningRate/len(miniBatch))*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(learningRate/len(miniBatch))*nb for b, nb in zip(self.biases, nabla_b)]

	def BackPropagate(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# Feed Forward
		activation = x
		activations = [x]
		zs = []
		
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b # was + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
				
		# Back Propagate
		delta = self.CostDerivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		
		return (nabla_b, nabla_w)
	
	def Evaluate(self, test_data):
		test_results = [(np.argmax(self.FeedForward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def CostDerivative(self, output_activations, y):
		return (output_activations-y)
