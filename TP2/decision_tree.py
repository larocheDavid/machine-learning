# Machine Learning - arbre de décision - David Laroche - 2019

import numpy as np


class Node:

	def __init__(self, data, maxDepth):
		self.left = None
		self.right = None
		self.leaf = False
		self.data = data
		self.maxDepth = maxDepth
		self.doneAttribute = [False] * 6
		self.attribute = None
		self.section = None
		self.gini = None
		self.compute()

	def printData(self):
		print(self.data)
	
	
	def oneAttribToClassMat(self, attribNumber):
		return self.data[:, [0, attribNumber]]


	def countClassPerSection(self, attribMat, sectionNumber):
		purClass0, impClass0, purClass1, impClass1 = 0, 0, 0, 0

		for row in attribMat:
			if row[1] <= sectionNumber:

				if row[0] == 0:
					purClass0 += 1
				else:
					impClass0 += 1
			else:
				if row[0] == 1:
					purClass1 += 1
				else:
					impClass1 += 1
				
		return (purClass0, impClass0, purClass1, impClass1) 


	def computeGini(self, results):

		purClass0, impClass0, purClass1, impClass1 = results[0], results[1], results[2], results[3]
		
		print(purClass0, impClass0, purClass1, impClass1)

		totalClass0 = purClass0 + impClass0
		totalClass1 = purClass1 + impClass1
		
		gNode1 = 1 - (purClass0/totalClass0)**2 - (impClass0/totalClass0)**2
		
		gNode2 = 1 - (purClass1/totalClass1)**2 - (impClass1/totalClass1)**2
		
		giniPond = totalClass0/(totalClass0+totalClass1)*gNode1 + totalClass1/(totalClass0+totalClass1)*gNode2

		
		return (gNode1, gNode2, giniPond)


	def findBestSection(self, attribMat):
		
		bestGini = 0.5
		best_section = 1
		
		lenAttribute = int(max(attribMat[:,1]))
		
		for sectionNumber in range(1, lenAttribute):
			
			classPerSection = self.countClassPerSection(attribMat, sectionNumber)
			
			if (classPerSection[0] + classPerSection[1] != 0 and classPerSection[2] + classPerSection[3] != 0):

				currGini = self.computeGini(classPerSection)
				print("curr_gini", currGini)
				print("Currsection", sectionNumber)
				
				if 	currGini[2] < bestGini:
					bestGini = currGini[2]
					best_section = sectionNumber
		
		return (bestGini, best_section)


	def findBestAttribute(self):
		bestGini = 0.5
		bestAttrib = 1

		for attribNumber in range(1, len(self.data[0])):
			if self.doneAttribute[attribNumber-1] == False:
				attribmat = self.oneAttribToClassMat(attribNumber)		
				attriBestSection = self.findBestSection(attribmat)
				print("currATtrib", attribNumber)
				if attriBestSection[0]  < bestGini:
					bestGini = attriBestSection[0]
					bestAttrib = attribNumber
					bestSection = attriBestSection[1]

		return (bestGini, bestAttrib, bestSection)


	def splitData(self):
		
		dataL, dataR = [], []

		for row in self.data:
			if row[self.attribute] <= self.section:
				dataL.append(row)		
			else:
				dataR.append(row)

		'''
		purClass0, impClass0, purClass1, impClass1 = 0, 0, 0, 0
		for row in dataL:
			if row[0] == 0:
				purClass0 += 1
			else:
				impClass0 += 1

		for row in dataR:
			if row[0] == 1:
				purClass1 += 1
			else:
				impClass1 += 1
		#print("GINI", self.computeGini((purClass0, impClass0, purClass1, impClass1)))		
		'''
	
		return (np.array(dataL), np.array(dataR))

	def compute(self):
		
		self.gini, self.attribute, self.section = self.findBestAttribute()

		self.doneAttribute[self.attribute-1] = True

		print("Attribute", self.attribute, "Section", self.section, "Gini", self.gini, "depth", self.maxDepth)
		
		dataL, dataR = self.splitData()


		#print("GINI", self.computeGini((5, 1, 2, 4)))
		#print(dataR)
		
		if self.left is None and len(set(dataL[:,0])) == 2:
			#print("anyL", dataL.any())
			#print("dataL", dataL)
			self.left = Node(dataL, self.maxDepth-1)
			
		if self.right is None and len(set(dataR[:,0])) == 2:
			print("DATAR\n", dataR)
			#print("anyR", dataR.any())
			#print("dataR", dataR)
			self.right = Node(dataR, self.maxDepth-1)
		


		'''
		if self.maxDepth > 0:
			print("depth", self.maxDepth)
			if self.left is None and len(set(dataL[:,0])) == 2:
				#print("anyL", dataL.any())
				#print("dataL", dataL)
				self.left = Node(dataL, self.maxDepth-1)
				
			if self.right is None and len(set(dataR[:,0])) == 2:
				#print("anyR", dataR.any())
				#print("dataR", dataR)
				self.right = Node(dataR, self.maxDepth-1)
		
		'''
	
def dataFileToMat(filename):
	return np.loadtxt(filename, usecols=(0,1,2,3,4,5,6))

max_depth = 5

dataTrain_1 = dataFileToMat("data-20191014/monks-1.train")

three = Node(dataTrain_1, max_depth)

