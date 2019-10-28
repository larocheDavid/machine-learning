# Machine Learning - arbre de décision - David Laroche - 2019

import numpy as np

'''
def dataFileToMat(filename):
	return  np.loadtxt(filename, usecols=(0,1,2,3,4,5,6))


def oneAttribToClassMat(dataMat, attribNumber):
	return dataMat[:, [0, attribNumber]]


def countClassPerSection(attribMat, sectionNumber):
	purClass0, purClass1 = 0, 0
	impClass0, impClass1 = 0, 0

	for row in attribMat:

		if row[1] <= sectionNumber:
			
			if row[0] == 0:
				purClass0 += 1
			else:
				impClass1 += 1

		else:
			if row[0] == 0:
				impClass0 += 1
			else:
				purClass1 += 1

	return (purClass0, purClass1, impClass0, impClass1) 


def gini(results):

	purClass0, purClass1, impClass0, impClass1 = results[0], results[1], results[2], results[3]
	
	totalClass0 = purClass0 + impClass0
	totalClass1 = purClass1 + impClass1
	
	gNode1 = 1 - (purClass0/totalClass0)**2 - (impClass0/totalClass0)**2
	
	gNode2= 1 - (purClass1/totalClass1)**2 - (impClass1/totalClass1)**2
	
	giniPond = (purClass0+impClass0)/(totalClass0+totalClass1)*gNode1 + (purClass1+impClass1)/(totalClass0+totalClass1)*gNode2

	return (gNode1, gNode2, giniPond)


def findBestSection(attribMat):
	
	bestGeany = 0.5
	best_section = 1
	
	lenAttribute = int(max(attribMat[:,1]))
	
	for sectionNumber in range(1, lenAttribute):
		currGeany = gini(countClassPerSection(attribMat, sectionNumber))
		print("gini", currGeany)
	
		if 	currGeany[2] < bestGeany:
			bestGeany = currGeany[2]
			best_section = sectionNumber
	
	return (bestGeany, best_section)


def findBestAttribute(dataMat):
	bestGeany = 0.5
	bestAttrib = 1

	for attribNumber in range(1, len(dataMat[0])):
		print(attribNumber)
		attribmat = oneAttribToClassMat(dataMat, attribNumber)		
		attriBestSection = findBestSection(attribmat)

		if attriBestSection[0]  < bestGeany:
			bestGeany = attriBestSection[0]
			bestAttrib = attribNumber
			bestSection = attriBestSection[1]

	return (bestGeany, bestAttrib, bestSection)
'''											 

#a = oneAttribToClassMat(dataTrain_1, 1)

#print(findBestAttribute(dataTrain_1))

class Node:

	def __init__(self, data, maxDepth):
		self.left = None
		self.right = None
		self.data = data
		self.maxDepth = maxDepth
		self.doneAttribute = [False] * 6
		self.compute()

	def printData(self):
		print(self.data)
	
	
	def oneAttribToClassMat(self, attribNumber):
		return self.data[:, [0, attribNumber]]


	def countClassPerSection(self, attribMat, sectionNumber):
		purClass0, purClass1 = 0, 0
		impClass0, impClass1 = 0, 0

		for row in attribMat:

			if row[1] <= sectionNumber:
				
				if row[0] == 0:
					purClass0 += 1
				else:
					impClass1 += 1

			else:
				if row[0] == 0:
					impClass0 += 1
				else:
					purClass1 += 1

		return (purClass0, purClass1, impClass0, impClass1) 


	def gini(self, results):

		purClass0, purClass1, impClass0, impClass1 = results[0], results[1], results[2], results[3]
		
		totalClass0 = purClass0 + impClass0
		totalClass1 = purClass1 + impClass1
		
		gNode1 = 1 - (purClass0/totalClass0)**2 - (impClass0/totalClass0)**2
		
		gNode2= 1 - (purClass1/totalClass1)**2 - (impClass1/totalClass1)**2
		
		giniPond = (purClass0+impClass0)/(totalClass0+totalClass1)*gNode1 + (purClass1+impClass1)/(totalClass0+totalClass1)*gNode2

		return (gNode1, gNode2, giniPond)


	def findBestSection(self, attribMat):
		
		bestGeany = 0.5
		best_section = 1
		
		lenAttribute = int(max(attribMat[:,1]))
		
		for sectionNumber in range(1, lenAttribute):
			currGeany = self.gini(self.countClassPerSection(attribMat, sectionNumber))
			print("gini", currGeany)
		
			if 	currGeany[2] < bestGeany:
				bestGeany = currGeany[2]
				best_section = sectionNumber
		
		return (bestGeany, best_section)


	def findBestAttribute(self):
		bestGeany = 0.5
		bestAttrib = 1

		for attribNumber in range(1, len(self.data[0])):
			print(attribNumber)
			attribmat = self.oneAttribToClassMat(attribNumber)		
			attriBestSection = self.findBestSection(attribmat)

			if attriBestSection[0]  < bestGeany:
				bestGeany = attriBestSection[0]
				bestAttrib = attribNumber
				bestSection = attriBestSection[1]

		return (bestGeany, bestAttrib, bestSection)

	def compute(self):
		#self.printData()
		print(self.findBestAttribute())




			
def dataFileToMat(filename):
	return np.loadtxt(filename, usecols=(0,1,2,3,4,5,6))

max_depth = 5

dataTrain_1 = dataFileToMat("data-20191014/monks-1.train")

three = Node(dataTrain_1, max_depth)

#root = Node(dataTrain_1, max_depth)
#root.printData()


#three.printData()


#print(countClassPerSection(a, 2))

#print("GINI", findBestSectionC0_C1(a))

#gin = (5, 2, 1, 4)
#print("GINI", gini(gin))
#print(findBestAttribute(dataTrain_1))


