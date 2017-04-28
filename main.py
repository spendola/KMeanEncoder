import numpy as np
from scipy import misc
import datetime as dt
import random
import kmean
import autoencoder
import datamanipulation
import publisher

featureSize = 12
colorChannels = 3
numberOfCentroids = 16

def main():
    print("Image Sharpening with Machine Learning\n")
    print("1) Train KMean Clustering")
    print("2) Train AutoEncoders")
    print("3) Enhance Image")
    print("4) Display Centroids")
    print("5) Test on Image")
    task = int(raw_input("\nChoose a Task: "))
    
    if task == 1:
        featureSize = int(raw_input("Feature Size: "))
        colorChannels = int(raw_input("Color Channels: "))
        numberOfCentroids = int(raw_input("Clusters: "))
        epochs = int(raw_input("Epochs: "))
        useLastCentroids = raw_input("Use last centroids? (y/n) ")
        startTime = dt.datetime.now().replace(microsecond=0)
        #images = ("images/Satellite_1.png", "C:/Images/Satellite_2.png", "C:/Images/Satellite_3.png", "C:/Images/Satellite_4.png", "C:/Images/Satellite_5.png")
        images = ("images/image_1.png", "images/image_2.png", "images/image_3.png", "images/image_4.png", "images/image_5.png", "images/image_6.png", "images/image_7.png", "images/image_8.png")
        data = datamanipulation.SliceImage(datamanipulation.LoadImageData(images), featureSize*colorChannels, featureSize)
        publisher.PublishMsg("KMean Clustering Started")
        publisher.PublishMsg("validationgraph")
        publisher.PublishMsg("flushgraph")
        KMean = kmean.KMean(featureSize*featureSize*colorChannels, numberOfCentroids)
        if(useLastCentroids == "y" or useLastCentroids == "Y"):
            data, centroids = KMean.KMeanClustering(data, datamanipulation.LoadCentroids("C:/Images/TempCentroids_" + str(colorChannels) + ".txt"), limit=epochs)
        else:
            data, centroids = KMean.KMeanClustering(data, limit=epochs)
        for centroid in range(0, len(centroids)):
            points = [x for x in data if x[-3] == centroid]
            print("Cluster " + str(centroid) + " has " + str(len(points)) + " points")
        print("Execution Time: " + str(dt.datetime.now().replace(microsecond=0) - startTime))
        datamanipulation.SaveCentroids(centroids, "C:/Images/TempCentroids_" + str(colorChannels) + ".txt")
        
    elif task == 2:
        learningRate = float(raw_input("Learning Rate: "))
        lmbda = float(raw_input("Lambda: "))
        epochs = int(raw_input("Epochs: "))
        featureSize = int(raw_input("Feature Size: "))
        colorChannels = int(raw_input("Color Channels: "))

        # Load Centroids and Initialize KMean module
        centroids = datamanipulation.LoadCentroids("C:/Images/TempCentroids_" + str(colorChannels) + ".txt")
        KMean = kmean.KMean(featureSize*featureSize*colorChannels, len(centroids))
    
        # Load points and classify them
        #images = ("C:/Images/Satellite_1.png", "C:/Images/Satellite_2.png", "C:/Images/Satellite_3.png", "C:/Images/Satellite_4.png", "C:/Images/Satellite_5.png")
        images = ("images/image_1.png", "images/image_2.png", "images/image_3.png", "images/image_4.png", "images/image_5.png", "images/image_6.png", "images/image_7.png", "images/image_8.png")
        data = datamanipulation.SliceImage(datamanipulation.LoadImageData(images), featureSize*colorChannels, featureSize)
        data = KMean.KmeanClassification(data, centroids)
        
        trainingCosts = []
        for centroid in range(0, len(centroids)):
            encoder = autoencoder.AutoEncoder([featureSize*featureSize*colorChannels, 100, featureSize*featureSize*colorChannels])
            encoder.LoadParameters("C:/Images/EncoderParameters_" + str(colorChannels) + "_" + str(centroid) + ".txt")
            trainingSet = datamanipulation.CreateTrainingSetFromData(data, centroids, centroid, threshold=False)
            if(len(trainingSet) > 0):
                print("Training Samples of size " + str(len(trainingSet[0])))
                savePath = "C:/Images/EncoderParameters_" + str(colorChannels) + "_" + str(centroid) + ".txt"
                trainingCosts.append(encoder.Train(trainingSet, learningRate, lmbda, epochs, len(trainingSet) if len(trainingSet) < 20 else 20, path=savePath))
            else:
                print("Training Set " + str(index) + " is empty")
                trainingCosts.append(-1)
        
        for index in range(0, len(centroids)):
            print("AutoEncoder for Feature " + str(index) + " has a learning cost of " + str(trainingCosts[index]))
                
    elif task == 3:
        featureSize = int(raw_input("Feature Size: "))
        colorChannels = int(raw_input("Color Channels: "))
        
        # Load Centroids and Initialize KMean module
        centroids = datamanipulation.LoadCentroids("C:/Images/TempCentroids_" + str(colorChannels) + ".txt")
        KMean = kmean.KMean(featureSize*featureSize*colorChannels, len(centroids))
        
        # Load reference and target images
        image_a = datamanipulation.LoadImageData(["C:/Images/Satellite_1.png"], colorChannels)
        image_b = datamanipulation.LoadImageData(["C:/Images/Satellite_1_LQ.png"], colorChannels)
        compare_1 = datamanipulation.CompareImages(image_a, image_b)
        
        # Slice reference image and classify slices
        data = datamanipulation.SliceImage(image_b, featureSize*colorChannels, featureSize)
        data = KMean.KmeanClassification(data, centroids)
        
        for centroid in range(0, len(centroids)):
            points = [x for x in data if x[-3] == centroid]
            if(len(points) > 0):
                encoder = autoencoder.AutoEncoder([featureSize*featureSize*colorChannels, 100, featureSize*featureSize*colorChannels])
                if(encoder.LoadParameters("C:/Images/EncoderParameters_" + str(colorChannels) + "_" + str(centroid) + ".txt")):
                    for point in points:
                        input = np.array(point[:-3]).reshape(144, 1)
                        output = encoder.FeedForward(input/256.0)*256.0
                        image_b = datamanipulation.InsertSubImage(image_b, output.flatten(), featureSize, colorChannels, point[-2], point[-1], reference=image_a)
                else:
                    print("unable to load encoder")
        
        compare_2 = datamanipulation.CompareImages(image_a, image_b)
        print("Initial Comparation: " + str(compare_1))
        print("Final Comparation: " + str(compare_2))
        
    elif task == 4:
        centroids = datamanipulation.LoadCentroids("C:/Images/TempCentroids.txt")
        datamanipulation.DisplayCentroids(centroids)
        
    elif task == 5:
        featureSize = int(raw_input("Feature Size: "))
        colorChannels = int(raw_input("Color Channels: "))
        centroids = datamanipulation.LoadCentroids("C:/Images/TempCentroids_" + str(colorChannels) + ".txt")
        datamanipulation.MarkImage("C:/Images/Satellite_2.png", centroids, featureSize, colorChannels)

        

if __name__ == '__main__':
    main()
    raw_input("press any key to exit")
