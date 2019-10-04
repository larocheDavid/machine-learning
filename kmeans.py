# David Laroche 2019

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import colors
import random
import math

class Cluster:

	def __init__(self, centroidXY, color):
		self.clusterPoints = [];
		self.centroidXY = centroidXY;
		self.color = color;
	
	def drawMarker(self, point, marker):
		plt.scatter(point[0], point[1], c=self.color, marker=marker)
		plt.pause(0.1)
		
	def clearMarker(self, point, marker):
		plt.scatter(point[0], point[1], c='w', marker=marker)
	
	def addPoint(self, point): # add point and draw it
		self.clusterPoints.append(point)
		self.drawMarker(point, marker='o')	

	def setCentroid(self, newPoint):
		self.centroidXY = newPoint
		self.drawMarker(newPoint, marker='+')

	def swapPoint(pos1, pos2, thatCluster):
		self.clusterPoints[pos1], thatCluster.clusterPoints[pos2] = self.clusterPoints[pos2], thatCluster.clusterPoints[pos1]
		
	def computeNewCentroid(self):
		self.clearMarker(self.centroidXY, marker='+')
		self.centroidXY = np.mean(self.clusterPoints, axis=0)
		self.drawMarker(self.centroidXY, marker='+')
	
	def computeVar(self):
		
		self.computeNewCentroid()
		var = 0
		for point in self.clusterPoints:
			var += pointDistance(point, self.centroidXY)
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
	return math.sqrt((A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]))


def findClosestPoint(cluster, dataList):
	distances = []
	for pointData in dataList:
		distances.append((pointDistance(pointData, cluster.centroidXY), pointData))
	
	closestPoint = sorted(distances)[0][1]
	dataList.remove(closestPoint)
	return closestPoint

def updateCentroids(clusterList):
	for cluster in clusterList:
		cluster.computeNewCentroid()

def swapPoints(pointA, pointB):
	pointA, pointB = pointB, pointA
		
numberOfCluster = 2
totalPoints =  60
domainLen = 60
timePause = 0.1

dataMat = np.zeros((domainLen, domainLen)) # initialise matrix

initDatas(dataMat, totalPoints) # initialise data in the matrix

dataList = coordinatesList(dataMat) # vector of data's coordinate

for point in dataList:
	plt.scatter(point[0], point[1], c='g')
plt.pause(timePause)

colorList = ['b', 'r', 'c', 'm', 'y', 'k']
clusterList = []
for i in range(numberOfCluster):
	centroidXY = putRandomPoint(dataMat, 2)
	clusterList.append(Cluster(centroidXY, colorList[i])) # make list of clusters
	clusterList[i].setCentroid(centroidXY)

for j in range(totalPoints//numberOfCluster):
	for i in range(numberOfCluster):
		closestPoint = findClosestPoint(clusterList[i], dataList)
		clusterList[i].addPoint(closestPoint)

var = totalPoints*domainLen

while True:
	for i in range(len(clusterList[0].clusterPoints)):
		for j in range(len(clusterList[1].clusterPoints)):
			swapPoint(clusterList[0].clusterPoints[i], clusterList[1].clusterPoints[j])

			varNew = clusterList[0].clusterPoints[i].computeVar + clusterList[1].clusterPoints[j].computeVar + 
			if varNew < var:
				var = varNew
				plt.scatter(clusterList[0].clusterPoints[i][0], clusterList[0].clusterPoints[i][1], c=clusterList[0].color, marker='o')
				plt.pause(1)
				plt.scatter(clusterList[1].clusterPoints[j][0], clusterList[1].clusterPoints[j][1], c=clusterList[1].color, marker='o')
				plt.pause(1)
				updateCentroids(clusterList)
				print("SWAP")
			else:
				swapPoint(clusterList[0].clusterPoints[i], clusterList[1].clusterPoints[j])
	
	break

print("Algo terminated")
plt.show()

