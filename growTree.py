import numpy as np
import pandas as pd
import json
import math


def calculateEntropy(attributeValue, allValues, labels):
    attributeCount = np.count_nonzero(allValues == attributeValue)

    classLabel1Count = 0
    classLabel2Count = 0
    
    for index, value in enumerate(allValues,0):
        if(value == attributeValue):
            #checking class label
            classLabel = labels[index]

            if(classLabel == 1):
                classLabel1Count = classLabel1Count + 1
            else:
                classLabel2Count = classLabel2Count + 1

    if(attributeCount == 0):
        return 0
    if(classLabel1Count == 0):
        return -(classLabel2Count/attributeCount) * math.log2(classLabel2Count/attributeCount)
    elif(classLabel2Count == 0):
        return -(classLabel1Count/attributeCount) * math.log2(classLabel1Count/attributeCount)
    else:
        return -(classLabel1Count/attributeCount) * math.log2(classLabel1Count/attributeCount) - (classLabel2Count/attributeCount) * math.log2(classLabel2Count/attributeCount)  


def calculateSetEntropy(trainingSet):
    
    classLabel1Count = 0
    classLabel2Count = 0

    totalSetCount = trainingSet.shape[1]
        
    #row with class labels in trainingSet
    classLabels = trainingSet[0]
    
    #calculating the occurence of each class label
    for index, value in enumerate(classLabels):
        if(value == 1):
            classLabel1Count = classLabel1Count + 1
        else:
            classLabel2Count = classLabel2Count + 1
                

    return -(classLabel1Count/totalSetCount) * math.log2(classLabel1Count/totalSetCount) - (classLabel2Count/totalSetCount) * math.log2(classLabel2Count/totalSetCount) 
    
        

def selectTestAttribute(data, description, hierarchy):

    maxGain = -1
    testAttribute = None
    
    entropyS = calculateSetEntropy(data)

    for index, value in enumerate(description):
        if(index not in hierarchy):
            #unique values for each attribute
            uniqueValues = value[1]

            #variable to store weighted average entropy
            weightedAvgEntropy = 0
            
            for attributeValue in uniqueValues:

                #all the class labels for customers
                labels = data[0]

                #row with attribute values
                allAttributeValues = data[index]

                #calculating entropy
                entropy = calculateEntropy(attributeValue,allAttributeValues,labels)

                #count of attribute value
                attributeCount = np.count_nonzero(allAttributeValues == attributeValue)

                numberOfColumns = data.shape[1]
                
                weightedAvgEntropy = weightedAvgEntropy + (entropy * (attributeCount/numberOfColumns))


            gain = entropyS - weightedAvgEntropy
            

            if(gain > maxGain):
                testAttribute = description[index]
                maxGain = gain           
               

    return testAttribute

def filterData(data, attributeValue, attributeIndex):

    #indexes to remove from data
    filteredIndexes = []
    
    #getting list of attributes
    attributeSet = data[attributeIndex]

    #getting indexes we want to remove
    for index, value in enumerate(attributeSet):
        #checking if value is attribute value
        if(value != attributeValue):
            filteredIndexes.append(index)
    

    return np.delete(data, filteredIndexes, axis=1)
    

def findIndex(description, attributeName):

    attributeIndex = 0
    
    for index, value in enumerate(description):
        name = value[0]
        if(name == attributeName):
            attributeIndex = index

    return attributeIndex

def maxOccurence(data):

    mostFrequentValue = data[0]
    classLabel1Count = 0
    classLabel2Count = 0

    for index, value in enumerate(data):
        if(value == 1):
            classLabel1Count = classLabel1Count + 1
        else:
            classLabel2Count = classLabel2Count + 1


    if(classLabel1Count > classLabel2Count):
        mostFrequentValue = 1
    else:
        mostFrequentValue = 2

    return mostFrequentValue
    


def generateTree(data, description, hierarchy, attributeValue, attributeIndex):

    #filter data
    filteredData = data
    currentLabels = []
    
    if(attributeValue != None): #will be none for root node
        filteredData = filterData(data, attributeValue, attributeIndex)
        #new class labels
        currentLabels = filteredData[0]

    #check for stopping conditions or leaf nodes
    if(attributeValue != None and filteredData.size == 0):
        classLabels = data[0]
        mostFreqLabel = maxOccurence(classLabels)
        return mostFreqLabel
    elif(attributeValue != None and np.all(currentLabels == currentLabels[0])):
        return int(currentLabels[0])
    else:
        #now we need to check for test attribute since we don't have stopping condition
        testAttribute = selectTestAttribute(filteredData, description, hierarchy)

        if(testAttribute == None): #we reach a point where we exhaust all divisions
            mostFreqLabel = maxOccurence(currentLabels)
            return mostFreqLabel
        else:               
            subTree = [testAttribute[0], {}]

            #index for test attribute so we can use to filter data 
            arrIndex = findIndex(description,testAttribute[0])

            #updating tree hierarchy
            hierarchy.append(arrIndex)
        
            branches = testAttribute[1]

            for branchValue in branches:
                value = generateTree(filteredData,description,hierarchy,branchValue,arrIndex)
                newDic = {str(branchValue) : value}
                treeDic = subTree[1]
                treeDic.update(newDic)
                subTree[1] = treeDic

            return subTree

#method to obtain label connected to feature/attribute value
def getLabel(value,feature,dataMap):

    #looping through data mappping
    for index, featureSet in enumerate(dataMap):
        keys = list(featureSet.keys())
        if(feature in keys):
            labelMapping = featureSet[feature]
            labelKeys = list(labelMapping.keys())
            labelValues = list(labelMapping.values())
            valuePosition = labelValues.index(value)
            correspondingLabel = labelKeys[valuePosition]
            return correspondingLabel

#method used to get feature/attribute name connected to label
def getFeatureName(value,feature,dataMap):

    #looping through mappings
    for index, featureSet in enumerate(dataMap):
        keys = list(featureSet.keys())
        if(feature in keys):
            labelMapping = featureSet[feature]
            labelKeys = list(labelMapping.keys())
            labelValues = list(labelMapping.values())
            valuePosition = labelKeys.index(value)
            return labelValues[valuePosition]
    
def printTree(decisionTree, level, dataMapping):

    node = decisionTree[0]
    space = " " * level
    print(space + node)

    branches = decisionTree[1]

    for key in branches:
        branchLevel = level + 2
        branchesSpace = " " * (branchLevel)
        #branch value
        name = getFeatureName(key,node,dataMapping)
        print(branchesSpace + str(name))

        leaf = branches[key]
        #checking if leaf is end
        isArr = isinstance(leaf, list)

        if(isArr):
            printTree(leaf,level+4,dataMapping)
        else:
            subBranchLevel = level + 4
            subBranchSpace = " " * subBranchLevel
            print(subBranchSpace + str(leaf))


def main():
    #getting trainingData
    trainingSet = pd.read_csv('trainingData.csv')
    headers = trainingSet.head(0)


    dataDescription = []
    dataMapping = []
    
    #creating data description and data label mapping array
    for row in headers:
        title = row
        
        #getting all unique values for column/feature
        uniqueValues = trainingSet[str(title)].unique()
        labelArr = []
        mappingDescription = {}
        
        #creating dictionary to map feature value to label
        for index, value in enumerate(uniqueValues):
            #label is index from 1 .. n-number of unique values
            label = index + 1
            labelArr.append(label)
            #feature value being mapped to its label
            mappingDescription.update({str(label) : str(value)})

        dataMapping.append({str(title) : mappingDescription})
        dataDescription.append([str(title), labelArr])


    print("Data Description Labels for each feature:")
    print(" ")
    print(dataDescription)
    print(" ")
    
    print("Label meanings for each feature:")
    print(" ")
    print(dataMapping)
    print(" ")
    
    #getting data shape(rows x columns)
    shapeArr = []
    shapeArr.append(trainingSet.shape[1])
    shapeArr.append(trainingSet.shape[0])

    #empty data array to store training set only using labels
    data = np.zeros(shapeArr)

    #create training data set which has only labels in data description
    for i, rowValue in trainingSet.iterrows():
        for index, feature in enumerate(dataDescription):
            featureName = feature[0]
            featureValue = rowValue[str(featureName)]
            label = getLabel(featureValue,featureName,dataMapping)
            data[index][i] = label

    

    #calling method to generate decision tree
    decisionTree = generateTree(data, dataDescription, [0], None, None)

    #printing decision tree
    printTree(decisionTree, 0, dataMapping)
    
    

main()
