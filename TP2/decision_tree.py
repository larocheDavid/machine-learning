# Machine Learning - arbre de décision - David Laroche - 2019

import numpy as np


class Node:

	def __init__(self, data, maxDepth, doneAttribute):
		self.left = None
		self.right = None
		self.leaf = False
		self.data = data
		self.maxDepth = maxDepth
		self.doneAttribute = doneAttribute
		self.attribute = None
		self.section = None
		self.gini = None
		self.compute()
	
	
	def oneAttribToClassMat(self, attribNumber):
		return self.data[:, [0, attribNumber]]


	def parentGini(self):
		c1, c2 = 0, 0
		
		for row in self.data:
			if row[0] == 0:
				c1 += 1
			else:
				c2 += 1

		return (c1, c2)


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
		#print(purClass0, impClass0, purClass1, impClass1)

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
				
				if 	currGini[2] < bestGini:
					bestGini = currGini[2]
					best_section = sectionNumber
		
		return (bestGini, best_section)


	def findBestAttribute(self):
		bestGini = 0.5
		bestAttrib, bestSection = None, None

		for attribNumber in range(1, len(self.data[0])):
			if self.doneAttribute[attribNumber-1] == False:
				attribmat = self.oneAttribToClassMat(attribNumber)
				
				attriBestSection = self.findBestSection(attribmat)

				#print("currATtrib", attribNumber)
				if attriBestSection[0]  < bestGini:
					bestGini = attriBestSection[0]
					bestAttrib = attribNumber
					bestSection = attriBestSection[1]
		
		if bestAttrib == None:
			#bestAttrib = self.attribute
			self.leaf = True

		return (bestGini, bestAttrib, bestSection)


	def splitData(self):
		
		dataL, dataR = [], []

		for row in self.data:
			if row[self.attribute] <= self.section:
				dataL.append(row)		
			else:
				dataR.append(row)

		return (np.array(dataL), np.array(dataR))



	def compute(self):
		
		print("Pgini", self.parentGini())
		self.gini, self.attribute, self.section = self.findBestAttribute()
		
		if self.leaf == True or self.maxDepth == 0:
			return

		else:
			

			self.doneAttribute[self.attribute-1] = True
			print("attr", self.attribute, "gini", self.gini, self.doneAttribute)


			#print("Attribute", self.attribute, "Section", self.section, "Gini", self.gini, "depth", self.maxDepth)
				
			dataL, dataR = self.splitData()
				
			#print("GINI", self.computeGini((5, 1, 2, 4)))
			if len(set(self.doneAttribute)) == 2:
				if self.left is None and len(set(dataL[:,0])) == 2:
					self.left = Node(dataL, self.maxDepth-1, self.doneAttribute)
						
				if self.right is None and len(set(dataR[:,0])) == 2:
					self.right = Node(dataR, self.maxDepth-1, self.doneAttribute)

	
def dataFileToMat(filename):
	return np.loadtxt(filename, usecols=(0,1,2,3,4,5,6))

max_depth = 100

dataTrain_1 = dataFileToMat("data-20191014/monks-1.train")

three = Node(dataTrain_1, max_depth, [False] * 6)

COUNT = [25]

def print2DUtil(three, space, max_depth) : 
  
    # Base case  
    if (three == None) : 
        return
  
    # Increase distance between levels  
    space += COUNT[0] 
  
    print2DUtil(three.left, space, max_depth)  
  
    # Print current node after space  
    # count  
    print()  
    for i in range(COUNT[0], space): 
        print(end = " ")  
    print("Level:", max_depth - three.maxDepth, "Attribute N°", three.attribute, "Gini", three.gini, three.doneAttribute)  
   
    print2DUtil(three.right, space, max_depth)  

print2DUtil(three, 0, max_depth)

