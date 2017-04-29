import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from mpl_toolkits.mplot3d import Axes3D
import publisher
import datamanipulation

class KMean():

    def __init__(self, dims, cens, dom=256):
        self.numberOfDimensions = dims
        self.numberOfCentroids = cens
        self.centroidDomain = dom
        print("KMean Module Initialized with " + str(self.numberOfDimensions) + " dimensions and " + str(self.numberOfCentroids) + " centroids")
    
    def GenerateCentroids(self):
        centroids = []
        for x in range(0, self.numberOfCentroids):
            temp = []
            for x in range(0, self.numberOfDimensions):
                temp.append(rnd.randrange(0, self.centroidDomain))
            centroids.append(temp)
        return centroids
    
    def KMeanClustering(self, data, centroids=[], epoch=0, limit=100):
        distances = []
        newcentroids = []
        
        # Check centroids
        if(len(centroids) == 0):
            print("Generating Random Centroids")
            centroids = self.GenerateCentroids()
            
        # Calculate distances from each point to all centroids
        for point in data:
            temp = []
            for centroid in centroids:
                temp.append(self.EuclideanDistance(point[:-3], centroid))
            point[-3] = temp.index(min(temp))

        # Reposition centroids
        emptyClusters = 0
        for centroid in centroids:
            temp = []
            points = [x for x in data if x[-3] == centroids.index(centroid)]
            if(len(points) > 0):
                for index in range(0, len(centroid)):
                    temp.append(np.mean([item[index] for item in points]))
                newcentroids.append(temp)
            else:
                print("repositioning centroid " + str(centroids.index(centroid)))
                repositionedCentroid = data[rnd.randrange(0, len(data))][:-3]
                newcentroids.append(repositionedCentroid)
                emptyClusters = emptyClusters + 1
        
        # Calculate distance between new and old centroids
        error = 0.0
        for a, b in zip(centroids, newcentroids):
            error = error + self.EuclideanDistance(a, b)
            
        # Check for Continue/Exit
        publisher.PublishData(error)
        print("epoch " + str(epoch) + " completed. (error: " + str(error) + ")")
        if(centroids != newcentroids and epoch < limit):
            return self.KMeanClustering(data, newcentroids, epoch+1, limit)
        else:
            publisher.PublishMsg("clustering completed")
            print("clustering completed")
            
        # Return finals centroids
        return data, centroids

    def KmeanClassification(self, data, centroids):
        print("KMean Classification in " + str(len(data)) + " vectors")
            
        # Calculate distances from each point to all centroids
        for point in data:
            temp = []
            for centroid in centroids:
                temp.append(self.EuclideanDistance(point[:-3], centroid))
            point[-3] = temp.index(min(temp))
            
        return data
    
    def EuclideanDistance(self, a, b):
        return dist.euclidean(a, b)
