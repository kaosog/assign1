import numpy as np

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

#We do experiment using DT

wine = datasets.load_wine()
#print(wine)
#firstTwoFeatures = wine.data[:, :2]
#print(firstTwoFeatures)

#Separating data and target
alldata = wine.data
alltarget = wine.target
print(alldata)
print(alltarget)

#Design: Data Division
#      		      Total	Train (80%)	Test (20%)
#	class 0-------- 59----------47--------------12
#	class 1-------- 71----------57--------------14
#	class 2-------- 48----------38--------------10


trainData = []
trainTarget = []

testData = []
testClasses = []
 
#Class 0 data separation
for i in range(0,47):
	trainData.append(alldata[i])
	trainTarget.append(alltarget[i])
for i in range(47,59):
	testData.append(alldata[i])
	testClasses.append(alltarget[i])

#Class 1 data separation
for i in range(59,116):
	trainData.append(alldata[i])
	trainTarget.append(alltarget[i])
for i in range(116,130):
	testData.append(alldata[i])
	testClasses.append(alltarget[i])

#Class 2 data separation
for i in range(130,168):
	trainData.append(alldata[i])
	trainTarget.append(alltarget[i])
for i in range(168,178):
	testData.append(alldata[i])
	testClasses.append(alltarget[i])

#print(trainData)
#print(trainTarget)
#print(testData)



#Create the DT Model

dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(trainData,trainTarget) # Training is done and the model is created
modelPredictions = dt.predict(testData) # Tesing 
print(modelPredictions)
print(testClasses) # compare this values with the predictions and calculate accuracy









