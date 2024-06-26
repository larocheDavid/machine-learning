# Machine Learning - arbre de décision - David Laroche - 2019
import numpy as np

class Node:
	def __init__(self, dataTrain, done_attribute, count):
		self.left = None
		self.right = None
		self.c1 = 0
		self.c2 = 0
		self.testC1 = 0
		self.testC2 = 0
		self.gini = None
		self.testGini = 0
		self.dataTrain = dataTrain
		self.dataTest = None
		self.count = count
		self.attribute = None
		self.section = None
		self.done_attribute = done_attribute
		self.C = None

def dataFileToMat(filename):
	return np.loadtxt(filename, usecols=(0,1,2,3,4,5,6))

def countC1C2(data):
		c1, c2 = 0, 0
		
		for row in data:
			if row[0] == 0:
				c1 += 1
			else:
				c2 += 1
		return c1, c2


def computeGini(c1, c2):
	
	return 1 - (c1 / (c1+c2)) ** 2 - (c2 / (c1+c2)) ** 2


def giniPond(n1c1, n1c2, n2c1, n2c2):

	cTotal = n1c1 + n1c2 + n2c1 + n2c2			
	
	giniN1 = computeGini(n1c1, n1c2)
	giniN2 = computeGini(n2c1, n2c2)

	return (n1c1+n1c2)/cTotal * giniN1 + (n2c1+n2c2)/cTotal * giniN2


def splitData(attribute, section, data):
	
	dataL, dataR = [], []

	for row in data:
		if row[attribute] <= section:
			dataL.append(row)		
		else:
			dataR.append(row)

	return (np.array(dataL), np.array(dataR))


def findBestChildGini(node):

	bestGini = node.gini
	node.attribute = "Leaf"
	dataNodeL, dataNodeR = [], []

	for currAttribute in range(1, len(node.dataTrain[0])):
		if node.done_attribute[currAttribute-1] == False:	

			for currSection in range(int(max(node.dataTrain[:,currAttribute]))):

				n1, n2 = splitData(currAttribute, currSection, node.dataTrain)
				
				n1c1, n1c2 = countC1C2(n1)
				n2c1, n2c2 = countC1C2(n2)			

				if (n1c1 + n1c2) != 0 and (n2c1 + n2c2) != 0:
					
					currPondGini = giniPond(n1c1, n1c2, n2c1, n2c2)

					if currPondGini < bestGini:
						bestGini = currPondGini
						node.attribute = currAttribute
						node.section = currSection
						dataNodeL, dataNodeR = n1, n2


	if node.attribute != "Leaf":
		node.done_attribute[node.attribute-1] = True

	return (dataNodeL, dataNodeR, node.done_attribute)


def makeNode(node, max_depth, min_class):
	
	node.c1, node.c2 = countC1C2(node.dataTrain)
	node.gini = computeGini(node.c1, node.c2)
	dataNodeL, dataNodeR, done_attribute = findBestChildGini(node)

	if max_depth <= node.count:
		node.attribute = "Leaf"
		return
	
	if len(dataNodeL) > min_class:
		node.left = Node(dataNodeL, done_attribute.copy(), node.count+1)
		makeNode(node.left, max_depth, min_class)


	if len(dataNodeR) > min_class:
		node.right = Node(dataNodeR, done_attribute.copy(), node.count+1)
		makeNode(node.right, max_depth, min_class)


def percentageTest(node):
	if node.testC1 + node.testC2 != 0:
		if node.C == "C1":
			return node.testC1/(node.testC1+node.testC2)*100
		else:
			return node.testC2/(node.testC1+node.testC2)*100
		
def meanPercentage(node):
	
	if node != None:
		if node.attribute == "Leaf":
			COUNTLEAF[0] += 1
			PERCENT[0] += percentageTest(node)
		meanPercentage(node.left)
		meanPercentage(node.right)
	else:
		return

def treeTest(node, dataTest):

	node.dataTest = dataTest
	node.testC1, node.testC2 = countC1C2(node.dataTest)
	#node.testGini = computeGini(node.testC1, node.testC2)

	if node.c1 > node.c2:
		node.C = "C1"
	else:
		node.C = "C2"

	if node.attribute != "Leaf":
		dataL, dataR = splitData(node.attribute, node.section, node.dataTest)
	else:
		return

	treeTest(node.left, dataL)
	treeTest(node.right, dataR)


def printTree(tree, space): 
   
    if tree == None: 
        return

    space += COUNT[0] 
  
    printTree(tree.left, space)  

    print()  
    for i in range(COUNT[0], space): 
        print(end = " ") 

    print("Level:", tree.count, "Attrib:", tree.attribute, "Class", tree.C)

    for i in range(COUNT[0], space): 
        print(end = " ")
    print("Train C1C2:", tree.c1, tree.c2, "Train gini:", round(tree.gini, 2))
    

    for i in range(COUNT[0], space): 
        print(end = " ")
    print("Test C1C2:", tree.testC1, tree.testC2, "percentage:", round(percentageTest(tree)), "%")#"Test gini:", round(tree.testGini, 2))

    printTree(tree.right, space)  


def computeAndDrawTree(dataTrainName, dataTestName, max_depth, min_class):
	
	dataTrain = dataFileToMat(dataTrainName)
	dataTest = dataFileToMat(dataTestName)

	root = Node(dataTrain, [False] * 6, 0)	
	makeNode(root, max_depth, min_class)
	treeTest(root, dataTest)
	printTree(root, 0)
	
	return root


COUNT = [25]
COUNTLEAF = [0]
PERCENT = [0]
max_depth, min_class = 3, 2

r1 = computeAndDrawTree("data-20191014/monks-1.train", "data-20191014/monks-1.test", max_depth, min_class)
#r2 = computeAndDrawTree("data-20191014/monks-2.train", "data-20191014/monks-2.test", max_depth, min_class)

meanPercentage(r1)
print("Percentage success:", round(PERCENT[0]/COUNTLEAF[0]), "%")

