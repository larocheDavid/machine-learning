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
	best_section = None				
	n1c1, n1c2, n2c1, n2c2 =  None, None, None, None
	dataNodeL, dataNodeR = [], []
	
	giniL, giniR = None, None

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
						giniL = computeGini(n1c1, n1c2)
						giniR = computeGini(n2c1, n2c2)

	if node.attribute != "Leaf":
		node.done_attribute[node.attribute-1] = True

	return (dataNodeL, dataNodeR, giniL, giniR, node.done_attribute)


def makeNode(node, max_depth, min_class):
	
	node.c1, node.c2 = countC1C2(node.dataTrain)
	node.gini = computeGini(node.c1, node.c2)
	dataNodeL, dataNodeR, giniL, giniR, done_attribute = findBestChildGini(node)

	if max_depth <= node.count:
		return
	
	if len(dataNodeL) > min_class:
		node.left = Node(dataNodeL, done_attribute.copy(), node.count+1)
		makeNode(node.left, max_depth, min_class)
		

	if len(dataNodeR) > min_class:
		node.right = Node(dataNodeR, done_attribute.copy(), node.count+1)
		makeNode(node.right, max_depth, min_class)


def treeTest(node, dataTest):

	node.dataTest = dataTest
	node.testC1, node.testC2 = countC1C2(node.dataTest)
	node.testGini = computeGini(node.testC1, node.testC2)

	if node.attribute == "Leaf":
		return

	dataL, dataR = splitData(node.attribute, node.section, node.dataTest)

	treeTest(node.left, dataL)
	treeTest(node.right, dataR)


def printTree(tree, space) : 
   
    if (tree == None) : 
        return

    space += COUNT[0] 
  
    printTree(tree.left, space)  

    print()  
    for i in range(COUNT[0], space): 
        print(end = " ") 

    print("Level:", tree.count, "Attrib:", tree.attribute)

    for i in range(COUNT[0], space): 
        print(end = " ")
    print("Train C1C2:", tree.c1, tree.c2, "Train gini:", round(tree.gini, 2))
    

    for i in range(COUNT[0], space): 
        print(end = " ")
    print("Test C1C2:", tree.testC1, tree.testC2, "Test gini:", round(tree.testGini, 2))

    printTree(tree.right, space)  


COUNT = [28]
max_depth = 10
min_class = 0
done_attribute = [False] * 6

dataTrain_1 = dataFileToMat("data-20191014/monks-1.train")
dataTest_1 = dataFileToMat("data-20191014/monks-1.test")

root = Node(dataTrain_1, done_attribute, 0)
makeNode(root, max_depth, min_class)
treeTest(root, dataTest_1)
printTree(root, 0)

dataTrain_2 = dataFileToMat("data-20191014/monks-2.train")
dataTest_2 = dataFileToMat("data-20191014/monks-2.test")

root2 = Node(dataTrain_2, done_attribute, 0)
makeNode(root2, max_depth, min_class)
treeTest(root2, dataTest_2)
printTree(root2, 0)

