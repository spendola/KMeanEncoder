import numpy as np
import random as rnd
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.spatial.distance as dist
from scipy import misc
from PIL import Image

def SaveCentroids(points, path):
	print("Saving Centroids")
	with open(path, "w+") as file:
		for point in points:
			for dimension in point:
				file.write(str(dimension) + ",")
			file.write("\n")
		file.close

def LoadCentroids(path):
	centroids = []
	with open(path, 'r') as file:
		data = file.readlines()
		for line in data:
			temp = []
			for point in [x.strip() for x in line.split(',')]:
				if(len(point) > 0):
					temp.append(float(point))
			centroids.append(temp)
		file.close
	print(str(len(centroids)) + " centroids loaded")
	return centroids
	
def DisplayCentroids(centroids):
	data = np.array([])
	for centroid in centroids:
		side = int(math.sqrt((len(centroid)/3)))
		tile = np.reshape(centroid, (side, side, 3)).astype(np.uint8)
		data = np.hstack([data, tile]) if data.size else tile
	plt.imshow(data, cmap = cm.Greys_r, interpolation='none')
	plt.show()


def CreateTrainingSetFromData(data, centroids, index, threshold=True):
	distances = []
	trainingSet = []
	for slice in [x for x in data if x[-3] == index]:
		distances.append((dist.euclidean(slice[:len(centroids[index])], centroids[index]), slice[-2], slice[-1]))
	if len(distances) < 1: return zip(trainingSet, trainingSet)
	lowerDeviation =  np.mean([x for x, y, z in np.array(distances)]) - np.std([x for x, y, z in np.array(distances)])
	
	trainingSet = [data[x][:-3] for x in [distances.index(x) for x in distances if (x[0] < lowerDeviation or not threshold)]]
	trainingSet = [np.reshape(x, (len(trainingSet[0]), 1)) for x in trainingSet]
	trainingSet = [x/256.0 for x in trainingSet]
	return zip(trainingSet, trainingSet)

def CalculateMinDistance(point, centroids):
	temp = []
	for centroid in centroids:
		temp.append(dist.euclidean(point[:len(centroid)], centroid))
	return temp.index(min(temp))
	
def LoadImage(path):
	image = misc.face()
	image = misc.imread(path)
	return image
	
def LoadImageData(images, channels=3):
	print("Loading images as RGB" if channels==3 else "Loading Images as GreyScale")
	data = np.array([])
	shapes = [[0, 0]]
	for image in images:
		print(image)
		colorImage = misc.face()
		colorImage = misc.imread(image)
		temp = FlatRGB(colorImage) if channels==3 else GreyScale(colorImage)
		print temp.shape
		shapes.append(temp.shape)
		data = np.vstack([data, temp]) if data.size else temp
	return data

def CloneImage(image):
	clonedImage = np.empty_like(image)	
	clonedImage[:] = image
	return clonedImage
	
def SliceImage(image, width, height):
	data = []
	for row in xrange(0, int(len(image)/height)):
		for col in xrange(0, int(len(image[row])/width)):
			temp = []
			for r in range(0, height):
				for c in range(0, width):
					temp.append(image[(row*height)+r][(col*width)+c])
			temp.append(-1)
			temp.append(row)
			temp.append(col)
			data.append(temp)
	print(str(len(image)) + "x" + str(len(image[0])) + " image sliced into " + str(len(data)) + " subimages")
	return data

def MarkImage(image, centroids, featureSize, colorChannels):
	originalImage = misc.face()
	originalImage = misc.imread(image)
	data = SliceImage(FlatRGB(originalImage) if colorChannels == 3 else GreyScale(originalImage), featureSize*colorChannels, featureSize)
	
	# Calculate Smallest Distance
	for point in data:
		point[-3] = CalculateMinDistance(point, centroids)
	
	for c in range(0, len(centroids)):	
		tempImage = CloneImage(originalImage)
		distances = []
		
		# calculate threshold per centroid
		for slice in [x for x in data if x[-3] == c]:
			distances.append((dist.euclidean(slice[:len(centroids[c])], centroids[c]), slice[-2], slice[-1]))
		print ("Cluster " + str(c) + " has " + str(len(distances)) + " associated points")
		
		if len(distances) > 0:
			distances = np.array(distances)
			threshold =  np.mean([x for x, y, z in distances]) - (np.std([x for x, y, z in distances]) * 0.5)	
			for slice in [x for x in distances if x[0] < threshold]:
				MarkSquare(tempImage, featureSize, slice[1], slice[2])
			Display(tempImage)

def GreyScale(image):
	grey = np.zeros((image.shape[0], image.shape[1]))
	for rownum in range(len(image)):
		for colnum in range(len(image[rownum])):
			grey[rownum][colnum] = WeightedAverage(image[rownum][colnum])
	return grey

def WeightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]
	
def FlatRGB(image):
	flatRgb = np.zeros((image.shape[0], image.shape[1]*3))
	for rownum in range(len(image)):
		for colnum in range(len(image[rownum])):
			flatRgb[rownum][colnum*3] = image[rownum][colnum][0]
			flatRgb[rownum][(colnum*3)+1] = image[rownum][colnum][1]
			flatRgb[rownum][(colnum*3)+2] = image[rownum][colnum][2]
	return flatRgb

def CopySquare(data, size, row, col):
	square = np.zeros((size, size))
	for c in range(0, size):
		for r in range(0, size):
			square[r][c] = data[row+r][col+c]
	return square
	
def MarkSquare(data, size, row, col):
	row = row*size
	col = col*size
	for i in range(0, size):
		data[row][col+i][0] = 255
		data[row+i][col][0] = 255
		data[row+size-1][col+i][0] = 255
		data[row+i][col+size-1][0] = 255
		
def InsertSubImage(image, subimage, size, channels, row, col, reference=None):
	print("Inserting SubImage in " + str(row) + ", " + str(col))
	prow = row*size
	pcol = col*size*channels
	for r in range(0, size):
		for c in range(0, size*channels):
			newValue = subimage[(r*size*channels)+c]
			if(reference != None):
				refValue = reference[prow+r][pcol+c]
				oriValue = image[prow+r][pcol+c]
				indicator = "+" if(abs(refValue - oriValue) > abs(refValue - newValue)) else "-"
				print("Original: " + str(oriValue) + ", New: " + str(newValue) + ", Reference: " + str(refValue) + " " + indicator)
			image[prow+r][pcol+c] = newValue
	return image

def CompareImages(image_1, image_2):
	identical = 0
	difference = 0.0
	for x, y in zip(image_1, image_2):
		for a, b in zip(x, y):
			if(a == b):
				identical = identical + 1
			else:
				difference = difference + abs(a - b)
	return difference, identical


	
def Display(data):
	plt.imshow(data, cmap = cm.Greys_r)
	plt.show()
	
def Version():
	print("Image Manipulation Version 0.1")
