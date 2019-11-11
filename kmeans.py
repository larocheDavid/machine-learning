# Machine Learning - Kmeans - David Laroche - 2019

import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Cluster:

	def __init__(self, centroidXY, color):
		self.clusterPoints = [];
		self.centroidXY = centroidXY;
		self.color = color;
	

	def drawMarker(self, point, marker):
		plt.scatter(point[0], point[1], c=self.color, marker=marker)
		plt.pause(0.5)
		

	def clearMarker(self, point, marker):
		plt.scatter(point[0], point[1], c='w', marker=marker)
	
	
	def addPoint(self, point): # add point and draw it
		self.clusterPoints.append(point)
		self.drawMarker(point, marker='o')	
		
	def resetClusterPoints(self):
		self.clusterPoints = []


	def computeNewCentroid(self):
		self.clearMarker(self.centroidXY, marker='x')
		self.centroidXY = np.mean(self.clusterPoints, axis=0)
		self.drawMarker(self.centroidXY, marker='x')
	

	def computeVar(self):
		
		mean = np.mean(self.clusterPoints, axis=0)	
		var = 0
		for point in self.clusterPoints:
			var += (point[0]-mean[0])**2 + (point[1]-mean[1])**2
		
		#print("npvar", np.var(self.clusterPoints), "\n")		
		return var/len(self.clusterPoints)


def putRandomPoint(matrix, value): # put a point in matrix randomly, return coordinate
	spaceLen = len(matrix[0])

	while True: 
		randomLine = np.random.randint(spaceLen)
		randomCol = np.random.randint(spaceLen)

		if matrix[randomLine][randomCol] == 0:
			matrix[randomLine][randomCol] = value
			return (randomLine, randomCol)


def initDatas(matrix, totalPoints): # put points randomly in matrix
	for i in range(totalPoints):
		putRandomPoint(matrix, 1)


def coordinatesList(matrix): # return list of point's coordinates
	dataList = []
	
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			if matrix[i][j] == 1:
				dataList.append((i,j))
	return dataList


def pointDistance(A, B): # return distance between A B
	return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)


def findNearestK(clusterList, dataList, colorList):
	for point in dataList:
		dist = []
		for i in range(len(clusterList)):
			dist += [(pointDistance(point, clusterList[i].centroidXY), i)]

		dist, index = sorted(dist)[0] # return nearest K
		clusterList[index].addPoint(point)


def computeAllCentroid(clusterList):
	for cluster in clusterList:
		cluster.computeNewCentroid()

def resetAllClusterPoints(clusterList):
	for cluster in clusterList:
		cluster.resetClusterPoints()

def meanClustersVariance(clusterList):
	res = 0
	for cluster in clusterList:
		res += cluster.computeVar()

	return res/len(clusterList)


def findClusters(clusterList, dataList, colorList):

	variance = 1000000
	while True:
		
		findNearestK(clusterList, dataList, colorList)
		
		computeAllCentroid(clusterList)
		
		newVariance = meanClustersVariance(clusterList)
		
		resetAllClusterPoints(clusterList)

		if newVariance == variance:
			break

		variance = newVariance
		print("mean variances minimisation:", round(variance))


numberOfCluster = 4
totalPoints = 60
domainLen = 120
variance = 1000000
colorList = ['b', 'r', 'c', 'm', 'y', 'k']

if __name__ == "__main__":

	dataMat = np.zeros((domainLen, domainLen)) # initialise matrix

	initDatas(dataMat, totalPoints) # initialise data in the matrix

	dataList = coordinatesList(dataMat) # vector of data's coordinate
	#dataList = [(0,0), (2,0)]

	for point in dataList:
		plt.scatter(point[0], point[1], c='g')
	
	clusterList = []
	for i in range(numberOfCluster):
		centroidXY = putRandomPoint(dataMat, 2)
		clusterList.append(Cluster(centroidXY, colorList[i])) # make list of clusters
		clusterList[i].drawMarker(centroidXY, marker='x')
	
	findClusters(clusterList, dataList, colorList)

	print("Algo terminated")
	plt.show()

