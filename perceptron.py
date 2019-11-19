# Perceptron simple - David Laroche - 2019

import numpy as np
import random


and_data = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]]
or_data =  [[0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
xor_data = [[0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]]


def dataFileToMat(filename):
    return np.loadtxt(filename, dtype = int, usecols=(0, 1, 2, 3, 4, 5, 6))


def binaryTransformation(dataFile):

    data = dataFileToMat(dataFile)
    resData = []
        
    d = np.array(data)

    input_len = []
    for i in range(len(data[0])):
        
        input_len.append(d[:,i].max())

    for listExample in data:

        newExample = [0] * sum(input_len[1:])
    
        for i, input in enumerate(listExample):

            if input != 0:
                newExample[sum(input_len[:i+1]) - 1 - input] = 1

            resData.append(newExample)    

    return resData


def initWeights(length, low, high):
    weights = []
    
    for _ in range(length):
        weights.append(random.uniform(low, high))

    return weights


def stepFunc(output, seuil, c0, c1): # step function activation 

    return c0 if output <= seuil else c1


def computeOutExample(example, weights):
    
    inputs = example[1:]

    out = np.dot(weights, [BIAIS] + inputs)       
    seuil = (C0 + C1) / 2

    return stepFunc(out, seuil, C0, C1)


def perceptron(data, learning_rate, limit):

    weights = initWeights(len(data[0]), -1, 1)

    quadError = 0
    for _ in range(limit):

        example = data[random.randint(0, len(data) -1)]
        
        t = example[0]

        inputs = [BIAIS] + example[1:]

        out = computeOutExample(example, weights)

        quadError += 1 / 2 * (t - out) ** 2

        if t != out:
            weights += learning_rate * (t - out) * np.array(inputs)
    
    print("     Quadratic error", quadError)
    
    return weights


def testData(data, weights):

    error = 0

    for example in data:
        t = example[0]

        out = computeOutExample(example, weights)

        if out != t:
            error += 1

    print("     Test succes rate", int((1-error/len(data))*100), "%")


C0, C1 = 0, 1
BIAIS = 1
limit = 5000
learnRate1 = 0.1

print("And:")
weightListAnd = perceptron(and_data, learnRate1, limit)
testData(and_data, weightListAnd)

print("\nOr:")
weightListOr = perceptron(or_data, learnRate1, limit)
testData(or_data, weightListOr)

print("\nXor:")
weightListXor = perceptron(xor_data, learnRate1, limit)
testData(xor_data, weightListXor)

monk_1_data_train = binaryTransformation("data-20191014/monks-1.train")
monk_1_data_test = binaryTransformation("data-20191014/monks-1.test")

print("\nMonk 1 - Train to Train")
weightsMonk1Train = perceptron(monk_1_data_train, learnRate1, limit)
testData(monk_1_data_train, weightsMonk1Train)

print("\nMonk 1 - Test to Test")
weightsMonk1Test = perceptron(monk_1_data_test, learnRate1, limit)
testData(monk_1_data_test, weightsMonk1Test)

