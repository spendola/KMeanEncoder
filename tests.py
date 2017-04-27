# Unit Test File for KMean-AutoEncoder Tree
import kmean
import autoencoder
import datamanipulation
import numpy as np

def main():
	print("UnitTest: K-Mean/AutoEncoder Network\n")
	while(True):
		print("0 - Exit")
		print("1 - Create Training Set From Data UnitTest")
		print("2 - Save/Load AutoEncoder Parameters UnitTest")
		print("3 - KMean Clustering UnitTest")
		print("4 - KMean Classification UnitTest")
		print("5 - Feed Forward UnitTests")
		print("6 - AutoEncoder Output UnitTest")
		print("7 - Mark Image UnitTest")
		print("8 - Compare Images UnitTest")
		print("9 - Insert SubImage UnitTest")

		selection = int(raw_input("\nSelect the UnitTest you wish to run: "))
		if(selection == 0):
			break
		elif(selection == 1):
			Assert(CreateTrainingSetFromData_UnitTest())
		elif(selection == 2):
			Assert(SaveLoadAutoEncoderParameters_UnitTest())
		elif(selection == 3):
			Assert(KMeanClustering_UnitTest())
		elif(selection == 4):
			Assert(KmeanClassification_UnitTest())
		elif(selection == 5):
			Assert(FeedForward_UnitTest())
		elif(selection == 6):
			Assert(AutoEncoderOutput_UnitTest())
		elif(selection == 7):
			Assert(MarkImage_UnitTest())
		elif(selection == 8):
			Assert(CompareImages_UnitTest())
		elif(selection == 9):
			Assert(InsertSubImage_UnitTest())
		
def Assert(response):
	print "Test Passed\n" if response else "Test Failed\n"

def CopySquares_UnitTest():
	print("Copying Square UnitTest")
	rows = 8
	cols = 10
	left = 3
	top = 3
	size = 4
	data = [[c+(r*cols) for c in range(cols)] for r in range(rows)]
	print("Copying square of size " + str(size) + " at " + str(left) + ", " + str(top))		
	square = datamanipulation.CopySquare(data, size, left, top)
	if(square[0][0] != data[top][left]):
		return False
	elif(square[size-1][size-1] != data[top+size-1][left+size-1]):
		return False
	return True


def CreateTrainingSetFromData_UnitTest():
	featureSize = 12
	colorChannels = 1
	
	# Load Centroids and Initialize KMean module
	centroids = datamanipulation.LoadCentroids("C:/Images/TempCentroids_" + str(colorChannels) + ".txt")
	KMean = kmean.KMean(featureSize*featureSize*colorChannels, len(centroids))
	
	# Load image and slice it
	images = ("C:/Images/Satellite_1.png", "C:/Images/Satellite_2.png", "C:/Images/Satellite_3.png", "C:/Images/Satellite_4.png", "C:/Images/Satellite_5.png")
	data = datamanipulation.SliceImage(datamanipulation.LoadImageData(images), featureSize*colorChannels, featureSize)
	
	# Classify points
	loadedPoints = len(data)
	data = KMean.KmeanClassification(data, centroids)
	
	print("Creating Training Sets with Threshold")
	for index in range(0, len(centroids)):
		trainingSet = datamanipulation.CreateTrainingSetFromData(data, centroids, index)
		if(len(trainingSet) > 0):
			print("Training Set for centroid " + str(index) + " has length " + str(len(trainingSet)))
		else:
			print("Training Set for centroid " + str(index) + " is empty")
	
	print("Creating Training Sets without Threshold")
	trainingPoints = 0
	for index in range(0, len(centroids)):
		trainingSet = datamanipulation.CreateTrainingSetFromData(data, centroids, index, threshold=False)
		trainingPoints = trainingPoints + len(trainingSet)
		if(len(trainingSet) > 0):
			print("Training Set for centroid " + str(index) + " has length " + str(len(trainingSet)))
		else:
			print("Training Set for centroid " + str(index) + " is empty")
	
	print("Loaded points vs training points: " + str(loadedPoints) + " / " + str(trainingPoints))
	if(loadedPoints == trainingPoints):
		return True
	return False
	
def SaveLoadAutoEncoderParameters_UnitTest():
	print("AutoEncoder: Save and Load Parameters Unit Test")
	encoder = autoencoder.AutoEncoder([435, 100, 435])
	initialParameters = encoder.GetParameters()
	encoder.SaveParameters("C:/Images/testParameters.txt")
	encoder.LoadParameters("C:/Images/testParameters.txt")
	finalParameters = encoder.GetParameters()
	for a, b in zip(initialParameters, finalParameters):
		for m, n in zip(a, b):
			for p, q in zip(m, n):
				if(p == q):
					return False
	return True
	
def AutoEncoderOutput_UnitTest():
	print("AutoEncoder Output UnitTest")
	inputSize = 144
	encoder = autoencoder.AutoEncoder([inputSize, 100, inputSize])
	input = np.random.rand(inputSize,1)
	output = encoder.FeedForward(input)
	if(output.shape[0] == inputSize):
		return True
	return False

def MarkImage_UnitTest():
	image = datamanipulation.LoadImage("C:/Images/Satellite_1.png")
	for r, c in zip(range(0, 10), range(0, 10)):
		datamanipulation.MarkSquare(image, 12, r, c)
	datamanipulation.Display(image)
	return True

def KmeanClassification_UnitTest():
	featureSize = 12
	colorChannels = 1
	
	# Load Centroids and Initialize KMean module
	centroids = datamanipulation.LoadCentroids("C:/Images/TempCentroids_" + str(colorChannels) + ".txt")
	KMean = kmean.KMean(featureSize*featureSize*colorChannels, len(centroids))
	
	# Load Image and Slice it
	image = datamanipulation.LoadImageData(["C:/Images/Satellite_1_LQ.png"])
	data = datamanipulation.SliceImage(image, featureSize*colorChannels, featureSize)
	originalClusters = [x[-3] for x in data]
	
	# Classify Images
	data = KMean.KmeanClassification(data, centroids)
	finalClusters = [x[-3] for x in data]
	
	for a, b in zip(originalClusters, finalClusters):
		if(a == b):
			print("One or more points were left unclassified")
			return False
	
	return True

def KMeanClustering_UnitTest():
	featureSize = 12
	colorChannels = 3
	numberOfCentroids = 12
	images = ("C:/Images/Satellite_1.png", "C:/Images/Satellite_2.png")
	data = datamanipulation.SliceImage(datamanipulation.LoadImageData(images), featureSize*colorChannels, featureSize)
	KMean = kmean.KMean(featureSize*featureSize*colorChannels, numberOfCentroids)
	data, centroids = KMean.KMeanClustering(data, KMean.GenerateCentroids())
	
	for index in range(0, numberOfCentroids):
		points = [x for x in data if x[-3] == index]
		if(len(points) < 1): return False
	
	return True
	
def FeedForward_UnitTest():
	featureSize = 12
	colorChannels = 1
	
	# Load Centroids and Initialize KMean module
	centroids = datamanipulation.LoadCentroids("C:/Images/TempCentroids_" + str(colorChannels) + ".txt")
	KMean = kmean.KMean(featureSize*featureSize*colorChannels, len(centroids))
	
	# Load image and slice it
	image_a = datamanipulation.LoadImageData(["C:/Images/Satellite_1.png"], colorChannels)
	image_b = datamanipulation.LoadImageData(["C:/Images/Satellite_1_LQ.png"], colorChannels)
	compare_1 = datamanipulation.CompareImages(image_a, image_a)
	compare_2 = datamanipulation.CompareImages(image_a, image_b)
	data = datamanipulation.SliceImage(image_b, featureSize*colorChannels, featureSize)
	loadedPoints = len(data)
	
	# Classify points
	data = KMean.KmeanClassification(data, centroids)
	
	# Sort points by centroid
	classifiedPoints = 0
	for centroid in range(0, len(centroids)):
		points = [x for x in data if x[-3] == centroid]
		classifiedPoints = classifiedPoints + len(points)
		encoder = autoencoder.AutoEncoder([featureSize*featureSize*colorChannels, 100, featureSize*featureSize*colorChannels])
		if(encoder.LoadParameters("C:/Images/EncoderParameters_" + str(colorChannels) + "_" + str(centroid) + ".txt")):
			print("feeding points into encoder")
			for point in points:
				output = encoder.FeedForward(np.array(point[:-3]).reshape(144, 1))
				image_b = datamanipulation.InsertSubImage(image_b, output.flatten(), featureSize, colorChannels, point[-2], point[-1], referenceImage=image_a)
		else:
			print("unable to load encoder")
		#	return False
		
	compare_3 = datamanipulation.CompareImages(image_a, image_b)
	print("Compare 1: " + str(compare_1))
	print("Compare 2: " + str(compare_2))
	print("Compare 3: " + str(compare_3))
	
	if(loadedPoints == classifiedPoints):
		return True
	return False
	
def CompareImages_UnitTest():
	image_a = datamanipulation.LoadImageData(["C:/Images/Satellite_1.png"])
	image_b = datamanipulation.LoadImageData(["C:/Images/Satellite_1_LQ.png"])
	image_c = datamanipulation.LoadImageData(["C:/Images/Satellite_1_VLQ.png"])
	
	compare_1 = datamanipulation.CompareImages(image_a, image_a)
	compare_2 = datamanipulation.CompareImages(image_a, image_b)
	compare_3 = datamanipulation.CompareImages(image_a, image_c)
	
	if(compare_1 > 0):
		return False
	elif(compare_2 <= compare_1):
		return False
	elif(compare_3 <= compare_2):
		return False
	return True
	
def InsertSubImage_UnitTest():
	colorChannels = 1
	featureSize = 12
	image_a = datamanipulation.LoadImageData(["C:/Images/Satellite_1.png"], colorChannels)
	image_b = datamanipulation.LoadImageData(["C:/Images/Blue.png"], colorChannels)
	data = datamanipulation.SliceImage(image_a, featureSize*colorChannels, featureSize)
	
	difference_original, identical_original = datamanipulation.CompareImages(image_a, image_b)
	image_b = datamanipulation.InsertSubImage(image_b, data[0][:-3], featureSize, colorChannels, data[0][-2], data[0][-1])
	difference_modified, identical_modified = datamanipulation.CompareImages(image_a, image_b)
	
	
	if(difference_modified >= difference_original):
		return False
	elif((identical_modified - identical_original) != (featureSize*featureSize)):
		return False
	return True
	
if __name__ == '__main__':
	main()
