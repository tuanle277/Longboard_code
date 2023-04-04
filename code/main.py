 # since the quantitative values of the wave frequency is arbitrary depending on the ride, so the frenquency alone would be inefficient


import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import timeit

from getFeatures import *
from getData import *
from getWindow import *
from dataTrans import *

import matplotlib.pyplot as plt
import pywt
import json

import datetime as dt

from collections import defaultdict, Counter


# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier



def main(method):
	plt.rcParams["figure.figsize"] = [25.00, 25.0]
	plt.rcParams["figure.autolayout"] = True
	theSeed = 50
	np.random.seed(theSeed)
	tf.random.set_seed(theSeed)

	hParams = {
		"datasetProportion": 1.0,
		"numEpochs": 100,
		"denseLayers": [400, 200, 100, 50, 14, 5],
		"valProportion": 0.1,
		"resultsName": "freq_test_results",
		"predictName": "freq_predictions",
		"percentName": "freq_Percentage",
		"waveletName": 	"rbio3.1"
	}

	models = [
		"10",
		"20_10",
		"50_10",
		"70_20",
		"100_50_10",
		"200_50_10",
		"300_50_10",
		"300_100_10",
		"300_200_10",
		"400_200_10",
		"300_200_100_10",
		"400_200_100_10",
		"450_400_300_200_10",
		"450_300_200_100_10"
	]

	method = method.split("_")
	if method[0] == "LSTM":
			print("Method run: LSTM")
			x_train, y_train = getPathTrainData()
			x_test, y_test = x_train[:int(0.25 * x_train.shape[0])], y_train[:int(0.25 * y_train.shape[0])]
			x_pred = getPathPredData()

			x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
			x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
			x_pred = x_pred.reshape((x_pred.shape[0], 1, x_pred.shape[1]))

	elif method[0] == "LSTMF":
			print("Method run: LSTMF")
			x_train, y_train = getFeaturesTrainData()
			x_test, y_test = x_train[:int(0.25 * x_train.shape[0])], y_train[:int(0.25 * y_train.shape[0])]
			x_pred = getFeaturesPredData()

			x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
			x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
			x_pred = x_pred.reshape((x_pred.shape[0], 1, x_pred.shape[1]))
	else:
		print("Method run: const")
		if method[1] == "LSTMF":
			x_train, y_train = getConFeaturesTrainData()
			x_test, y_test = x_train[:int(0.25 * x_train.shape[0])], y_train[:int(0.25 * y_train.shape[0])]
			x_pred = getConFeaturesPredData()

			x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
			x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
			x_pred = x_pred.reshape((x_pred.shape[0], 1, x_pred.shape[1]))

		else:
			x_train, y_train = getConFeaturesTrainData()
			x_test, y_test = x_train[:int(0.25 * x_train.shape[0])], y_train[:int(0.25 * y_train.shape[0])]
			x_pred = getConFeaturesPredData()


	print(x_train.shape, y_train.shape)
	print(x_test.shape, y_test.shape)
	# x_train = feature_extraction(x_train, hParams['waveletName'])
	# x_test = feature_extraction(x_test, hParams['waveletName'])

	# x_train = feature_extraction(x_train, hParams['waveletName'])
	# x_test = feature_extraction(x_test, hParams['waveletName'])
	# x_pred = feature_extraction(x_pred, hParams['waveletName'])
	dataSubsets = (x_train, y_train, x_test, y_test, x_pred)


	for model in models:
		hParams = getHParams(model)
		trainResults, testResults, predictions = learn(dataSubsets, hParams)
		writeExperimentalResults(hParams, trainResults, testResults, predictions)

# ================================================================
# ================================================================
def writeExperimentalResults(hParams, trainResults, testResults, predictions):
	f = open("../results/models/LSTMF/" + hParams["resultsName"] + "_LSTMF" + ".txt", 'w')
	f.write(str(hParams) + '\n\n')
	f.write(str(trainResults) + '\n\n')
	f.write(str(testResults) + '\n\n')

	f1 = open("../results/models/LSTMF/" + hParams["predictName"] + "_LSTMF" + ".txt", 'w')
	f1.write(str(hParams) + '\n\n')
	f1.write(str(predictions))

	percentReport = prediction_result(predictions)
	writePercentReport(hParams, percentReport)

	f.close()
	f1.close()

def writePercentReport(hParams, percentReport):
	ff = open("../results/models/LSTMF/" + hParams["percentName"] + "_LSTMF" + ".txt", 'w')
	ff.write(str(percentReport))
	ff.close()

def writePredictions(hParams, predictions):
	ff = open("../results/models/LSTMF/" + hParams["predictName"] + "_LSTMF" + ".txt", 'w')
	ff.write(str(predictions))
	ff.close()

def readExperimentalResults(fileName):
	f = open("../results/models/LSTMF/" + fileName + "_LSTMF" + ".txt",'r')
	data = f.read().split('\n\n')

	data[0] = data[0].replace("\'", "\"")
	data[1] = data[1].replace("\'", "\"")

	hParams = json.loads(data[0])
	trainResults = json.loads(data[1])
	testResults = json.loads(data[2])

	return hParams, trainResults, testResults

def readPredResults(fileName):
	f = open("../results/models/LSTMF/" + fileName + "_LSTMF" + ".txt", "r")
	data = f.read().split('\n\n')

	data[0] = data[0].replace("\'", "\"")
	data[1] = data[1].replace("\'", "\"")

	hParams = json.loads(data[0])
	predictions = json.loads(data[1])
	return hParams, predictions

def plotPredictions(predictions, title):
	plotCurves(x=np.array([x for x in range(len(predictios))]), 
				yList= predictions, 
				xLabel="time (t)", 
				yLabelList=itemsToPlot, 
				title= title + "predictions")

def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
	fig, ax = plt.subplots()
	y = np.array(yList).transpose()
	ax.plot(x, y)
	ax.set(xlabel=xLabel, title=title)
	plt.legend(yLabelList, loc='best', shadow=True)
	ax.grid()
	yLabelStr = "__" + "__".join([label for label in yLabelList])
	filepath = "../results/models/LSTMF/" + title + " " + yLabelStr + ".png"
	fig.savefig(filepath)
	print("Figure saved in", filepath)

def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
	plt.figure()
	plt.scatter(xList,yList)
	plt.xlabel(xLabel)
	plt.ylabel(yLabel)
	plt.title(title)
	if pointLabels != []:
		for i, label in enumerate(pointLabels):
			plt.annotate(label, (xList[i], yList[i]))
	filepath = "../results/models/LSTMF/" + filename + ".png"
	plt.savefig(filepath)
	print("Figure saved in", filepath)


# ===========================================================================================================================
# ===========================================================================================================================

# def writeExperimentalResults(hParams, trainResults, testResults, predictions):
# 	f = open("results/models/LSTM/" + hParams["resultsName"] + "_LSTM" + ".txt", 'w')
# 	f.write(str(hParams) + '\n\n')
# 	f.write(str(trainResults) + '\n\n')
# 	f.write(str(testResults) + '\n\n')

# 	f1 = open("results/models/LSTM/" + hParams["predictName"] + "_LSTM" + ".txt", 'w')
# 	f1.write(str(hParams) + '\n\n')
# 	f1.write(str(predictions))

# 	percentReport = prediction_result(predictions)
# 	writePercentReport(hParams, percentReport)

# 	f.close()
# 	f1.close()

# def writePercentReport(hParams, percentReport):
# 	ff = open("results/models/LSTM/" + hParams["percentName"] + "_LSTM" + ".txt", 'w')
# 	ff.write(percentReport)
# 	ff.close()

# def writePredictions(hParams, predictions):
# 	ff = open("results/models/LSTM/" + hParams["predictName"] + "_LSTM" + ".txt", 'w')
# 	ff.write(str(predictions))
# 	ff.close()

# def readExperimentalResults(fileName):
# 	f = open("results/models/LSTM/" + fileName + ".txt",'r')
# 	data = f.read().split('\n\n')

# 	data[0] = data[0].replace("\'", "\"")
# 	data[1] = data[1].replace("\'", "\"")

# 	hParams = json.loads(data[0])
# 	trainResults = json.loads(data[1])
# 	testResults = json.loads(data[2])

# 	return hParams, trainResults, testResults

# def readPredResults(fileName):
# 	f = open("results/models/LSTM/" + fileName + ".txt", "r")
# 	data = f.read().split('\n\n')

# 	data[0] = data[0].replace("\'", "\"")
# 	data[1] = data[1].replace("\'", "\"")

# 	hParams = json.loads(data[0])
# 	predictions = json.loads(data[1])
# 	return hParams, predictions

# def plotPredictions(predictions, title):
# 	plotCurves(x=np.array([x for x in range(len(predictios))]), 
# 				yList= predictions, 
# 				xLabel="time (t)", 
# 				yLabelList=itemsToPlot, 
# 				title= title + "predictions")

# def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
# 	fig, ax = plt.subplots()
# 	y = np.array(yList).transpose()
# 	ax.plot(x, y)
# 	ax.set(xlabel=xLabel, title=title)
# 	plt.legend(yLabelList, loc='best', shadow=True)
# 	ax.grid()
# 	yLabelStr = "__" + "__".join([label for label in yLabelList])
# 	filepath = "results/models/LSTM/" + title + " " + yLabelStr + ".png"
# 	fig.savefig(filepath)
# 	print("Figure saved in", filepath)

# def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
# 	plt.figure()
# 	plt.scatter(xList,yList)
# 	plt.xlabel(xLabel)
# 	plt.ylabel(yLabel)
# 	plt.title(title)
# 	if pointLabels != []:
# 		for i, label in enumerate(pointLabels):
# 			plt.annotate(label, (xList[i], yList[i]))
# 	filepath = "results/models/LSTM/" + filename + ".png"
# 	plt.savefig(filepath)
# 	print("Figure saved in", filepath)

# ===========================================================================================================================
# ===========================================================================================================================

# def writeExperimentalResults(hParams, trainResults, testResults, predictions):
# 	f = open("results/models/features/" + hParams["resultsName"] + "_features" + ".txt", 'w')
# 	f.write(str(hParams) + '\n\n')
# 	f.write(str(trainResults) + '\n\n')
# 	f.write(str(testResults) + '\n\n')

# 	f1 = open("results/models/features/" + hParams["predictName"] + "_features" + ".txt", 'w')
# 	f1.write(str(hParams) + '\n\n')
# 	f1.write(str(predictions))

# 	percentReport = prediction_result(predictions)
# 	writePercentReport(hParams, percentReport)

# 	f.close()
# 	f1.close()

# def writePercentReport(hParams, percentReport):
# 	ff = open("results/models/features/" + hParams["percentName"] + ".txt", 'w')
# 	ff.write(percentReport)
# 	ff.close()

# def writePredictions(hParams, predictions):
# 	ff = open("results/models/features/" + hParams["predictName"] + ".txt", 'w')
# 	ff.write(str(predictions))
# 	ff.close()

# def readExperimentalResults(fileName):
# 	f = open("results/models/features/" + fileName + ".txt",'r')
# 	data = f.read().split('\n\n')

# 	data[0] = data[0].replace("\'", "\"")
# 	data[1] = data[1].replace("\'", "\"")

# 	hParams = json.loads(data[0])
# 	trainResults = json.loads(data[1])
# 	testResults = json.loads(data[2])

# 	return hParams, trainResults, testResults

# def readPredResults(fileName):
# 	f = open("results/models/features/" + fileName + ".txt", "r")
# 	data = f.read().split('\n\n')

# 	data[0] = data[0].replace("\'", "\"")
# 	data[1] = data[1].replace("\'", "\"")

# 	hParams = json.loads(data[0])
# 	predictions = json.loads(data[1])
# 	return hParams, predictions

# def plotPredictions(predictions, title):
# 	plotCurves(x=np.array([x for x in range(len(predictios))]), 
# 				yList= predictions, 
# 				xLabel="time (t)", 
# 				yLabelList=itemsToPlot, 
# 				title= title + "predictions")

# def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
# 	fig, ax = plt.subplots()
# 	y = np.array(yList).transpose()
# 	ax.plot(x, y)
# 	ax.set(xlabel=xLabel, title=title)
# 	plt.legend(yLabelList, loc='best', shadow=True)
# 	ax.grid()
# 	yLabelStr = "__" + "__".join([label for label in yLabelList])
# 	filepath = "results/models/features/" + title + " " + yLabelStr + ".png"
# 	fig.savefig(filepath)
# 	print("Figure saved in", filepath)

# def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
# 	plt.figure()
# 	plt.scatter(xList,yList)
# 	plt.xlabel(xLabel)
# 	plt.ylabel(yLabel)
# 	plt.title(title)
# 	if pointLabels != []:
# 		for i, label in enumerate(pointLabels):
# 			plt.annotate(label, (xList[i], yList[i]))
# 	filepath = "results/models/features/" + filename + ".png"
# 	plt.savefig(filepath)
# 	print("Figure saved in", filepath)

# ===========================================================================================================================
# ===========================================================================================================================

# def writeExperimentalResults(hParams, trainResults, testResults, predictions):
# 	f = open("../results/constt/features/" + hParams["resultsName"]  + ".txt", 'w')
# 	f.write(str(hParams) + '\n\n')
# 	f.write(str(trainResults) + '\n\n')
# 	f.write(str(testResults) + '\n\n')

# 	f1 = open("../results/constt/features/" + hParams["predictName"] + ".txt", 'w')
# 	f1.write(str(hParams) + '\n\n')
# 	f1.write(str(predictions))

# 	percentReport = prediction_result(predictions)
# 	writePercentReport(hParams, percentReport)

# 	f.close()
# 	f1.close()

# def writePercentReport(hParams, percentReport):
# 	ff = open("../results/constt/features/" + hParams["percentName"] + ".txt", 'w')
# 	ff.write(str(percentReport))
# 	ff.close()

# def writePredictions(hParams, predictions):
# 	ff = open("../results/constt/features/" + hParams["predictName"] + ".txt", 'w')
# 	ff.write(str(predictions))
# 	ff.close()

# def readExperimentalResults(fileName):
# 	f = open("../results/constt/features/" + fileName + ".txt",'r')
# 	data = f.read().split('\n\n')

# 	data[0] = data[0].replace("\'", "\"")
# 	data[1] = data[1].replace("\'", "\"")

# 	hParams = json.loads(data[0])
# 	trainResults = json.loads(data[1])
# 	testResults = json.loads(data[2])

# 	return hParams, trainResults, testResults

# def readPredResults(fileName):
# 	f = open("../results/constt/features/" + fileName + ".txt", "r")
# 	data = f.read().split('\n\n')

# 	data[0] = data[0].replace("\'", "\"")
# 	data[1] = data[1].replace("\'", "\"")

# 	hParams = json.loads(data[0])
# 	predictions = json.loads(data[1])
# 	return hParams, predictions

# def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
# 	fig, ax = plt.subplots()
# 	y = np.array(yList).transpose()
# 	ax.plot(x, y)
# 	ax.set(xlabel=xLabel, title=title)
# 	plt.legend(yLabelList, loc='best', shadow=True)
# 	ax.grid()
# 	yLabelStr = "__" + "__".join([label for label in yLabelList])
# 	filepath = "../results/constt/features/" + title + " " + yLabelStr + ".png"
# 	fig.savefig(filepath)
# 	print("Figure saved in", filepath)

# def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
# 	plt.figure()
# 	plt.scatter(xList,yList)
# 	plt.xlabel(xLabel)
# 	plt.ylabel(yLabel)
# 	plt.title(title)
# 	if pointLabels != []:
# 		for i, label in enumerate(pointLabels):
# 			plt.annotate(label, (xList[i], yList[i]))
# 	filepath = "../results/constt/features/" + filename + ".png"
# 	plt.savefig(filepath)
	# print("Figure saved in", filepath)

# ===============================================================
# =================================================================

def processResults():
	hParams, trainResults, testResults = readExperimentalResults("results")
	hParams, dPredictions = readPredResults("predictions")

	itemsToPlot = ['accuracy', 'val_accuracy']
	plotCurves(x=np.arange(0, hParams['numEpochs']), 
				yList=[trainResults[item] for item in itemsToPlot], 
				xLabel="Epoch",
				yLabelList=itemsToPlot, 
				title=hParams['resultsName'])

	itemsToPlot = ['loss', 'val_loss']
	plotCurves(x=np.arange(0, hParams['numEpochs']), 
				yList=[trainResults[item] for item in itemsToPlot], 
				xLabel="Epoch", 
				yLabelList=itemsToPlot, 
				title=hParams['resultsName'])

	writePredictions(hParams, dPredictions)
	percentReport = prediction_result(dPredictions)
	writePercentReport(hParams, percentReport)


def processResults():
	hParams, trainResults, testResults = readExperimentalResults("results")
	hParams, dPredictions = readPredResults("predictions")

	itemsToPlot = ['accuracy', 'val_accuracy']
	plotCurves(x=np.arange(0, hParams['numEpochs']), 
				yList=[trainResults[item] for item in itemsToPlot], 
				xLabel="Epoch",
				yLabelList=itemsToPlot, 
				title=hParams['resultsName'])

	itemsToPlot = ['loss', 'val_loss']
	plotCurves(x=np.arange(0, hParams['numEpochs']), 
				yList=[trainResults[item] for item in itemsToPlot], 
				xLabel="Epoch", 
				yLabelList=itemsToPlot, 
				title=hParams['resultsName'])

	writePredictions(hParams, dPredictions)
	percentReport = prediction_result(dPredictions)
	writePercentReport(hParams, percentReport)

def buildOverallResults(fileNames, title, folderName):
    # == get hParams == #
    hParams = readExperimentalResults(fileNames[0]+"_results")[0]

    # == plot curves with yList being the validation accuracies == #
    plotCurves(x = np.arange(0, hParams["numEpochs"]), 
            yList=[readExperimentalResults(name+"_results")[1]['val_accuracy'] for name in fileNames], 
            xLabel="Epoch",
            yLabelList=fileNames,
            title= "val_" + title)

    plotCurves(x = np.arange(0, hParams["numEpochs"]), 
            yList=[readExperimentalResults(name+"_results")[1]['accuracy'] for name in fileNames], 
            xLabel="Epoch",
            yLabelList=fileNames,
            title= "acc_" + title)

    # == plot points with xList being the parameter counts of all and yList being the test accuracies == #
    plotPoints(xList=[readExperimentalResults(name+"_results")[0]['paramCount'] for name in fileNames],
                yList=[readExperimentalResults(name+"_results")[2][0] for name in fileNames],
                pointLabels= [name for name in fileNames],
                xLabel='Number of parameters',
                yLabel='Test set loss',
                title="Test set loss_" + title,
                filename="Test set loss_" + title)

    # == plot points with xList being the parameter counts of all and yList being the test accuracies == #
    plotPoints(xList=[readExperimentalResults(name+"_results")[0]['paramCount'] for name in fileNames],
                yList=[readExperimentalResults(name+"_results")[2][1] for name in fileNames],
                pointLabels= [name for name in fileNames],
                xLabel='Number of parameters',
                yLabel='Test set acc',
                title="Test set acc_" + title,
                filename="Test set acc_" + title)

    fig, axs = plt.subplots(len(fileNames))
    index = 0
    for fileName in fileNames:
    	plt.figure()
    	predByModels = readPredResults(fileName+"_predict")
    	axs[index].plot([x for x in range(len(predByModels[1]))], [x for x in predByModels[1]])
    	axs[index].set_title(fileName)
    	index += 1

    filepath = "../results/constt/" + folderName + "/predictions.png"
    fig.savefig(filepath, dpi=200)
    print("Figure saved in", filepath)

def plotPredictions(y_test, predictions, title):
	# plotCurves(x=np.array([x for x in range(len(predictios))]), 
	# 			yList= predictions, 
	# 			xLabel="time (t)", 
	# 			yLabelList=itemsToPlot, 
	# 			title= title + "predictions")

    train = y_test
    train = y_test[:int(len(y_test)-len(predictions))]
    valid = y_test[int(len(y_test)-len(predictions)):]
    #visualize the data
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('time', fontsize=18)
    plt.ylabel('activity', fontsize=18)
    plt.plot(train)
    plt.plot(valid)
    plt.plot(predictions)
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig(title+"prediction")
    # plt.show()



    # train = df['Close']
    # train = df[:int(len(df)-len(predictions))]
    # valid = df[int(len(df)-len(predictions)):]
    # valid['Predictions'] = predictions
    # #visualize the data
    # plt.figure(figsize=(16,6))
    # plt.title('Model')
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Close Price USD ($)', fontsize=18)
    # plt.plot(train['Close'])
    # plt.plot(valid[['Close', 'Predictions']])
    # plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    # plt.show()

    # hParams = readExperimentalResults(fileNames[0]+"_results_LSTM")[0]

    # # == plot curves with yList being the validation accuracies == #
    # plotCurves(x = np.arange(0, hParams["numEpochs"]), 
    #         yList=[readExperimentalResults(name+"_results_LSTM")[1]['val_accuracy'] for name in fileNames], 
    #         xLabel="Epoch",
    #         yLabelList=fileNames,
    #         title= "val_LSTM_" + title)

    # plotCurves(x = np.arange(0, hParams["numEpochs"]), 
    #         yList=[readExperimentalResults(name+"_results_LSTM")[1]['accuracy'] for name in fileNames], 
    #         xLabel="Epoch",
    #         yLabelList=fileNames,
    #         title= "acc_LSTM_" + title)

    # # == plot points with xList being the parameter counts of all and yList being the test accuracies == #
    # plotPoints(xList=[readExperimentalResults(name+"_results_LSTM")[0]['paramCount'] for name in fileNames],
    #             yList=[readExperimentalResults(name+"_results_LSTM")[2][0] for name in fileNames],
    #             pointLabels= [name for name in fileNames],
    #             xLabel='Number of parameters',
    #             yLabel='Test set loss',
    #             title="Test set loss_LSTM_" + title,
    #             filename="Test set loss_LSTM_" + title)

    # # == plot points with xList being the parameter counts of all and yList being the test accuracies == #
    # plotPoints(xList=[readExperimentalResults(name+"_results_LSTM")[0]['paramCount'] for name in fileNames],
    #             yList=[readExperimentalResults(name+"_results_LSTM")[2][1] for name in fileNames],
    #             pointLabels= [name for name in fileNames],
    #             xLabel='Number of parameters',
    #             yLabel='Test set acc',
    #             title="Test set acc_LSTM_" + title,
    #             filename="Test set acc_LSTM_" + title)

    # fig, axs = plt.subplots(len(fileNames))
    # index = 0
    # for fileName in fileNames:
    # 	plt.figure()
    # 	predByModels = readPredResults(fileName+"_predict_LSTM")
    # 	axs[index].plot([x for x in range(len(predByModels[1]))], [x for x in predByModels[1]])
    # 	axs[index].set_title(fileName)
    # 	index += 1

    # filepath = "results/models/LSTM/predictions_LSTM.png"
    # fig.savefig(filepath, dpi=200)
    # print("Figure saved in", filepath)


def prediction_result(predictions):
	pred = {"pumping": 0, "pushing": 0, "coasting": 0, "carving": 0}
	for p in predictions:
		pred[p] += 1

	percentReport = "Percentage of pumping: " + "{:.1f}%".format(100 * pred["pumping"]/len(predictions), "%"),
	'\n' + "Percentage of pushing: " + "{:.1f}%".format(100 * pred["pushing"]/len(predictions), "%"),
	'\n' + "Percentage of coasting: " + "{:.1f}%".format(100 * pred["coasting"]/len(predictions), "%",
	'\n' + "Percentage of carving: " + "{:.1f}%".format(100 * pred["carving"]/len(predictions), "%"))
	return percentReport

def getHParams(expName=None):
	# Set up what's the same for each experiment
	hParams = {
		'experimentName': expName,
		'datasetProportion': 1.0,
		'valProportion': 0.1,
		'numEpochs': 100
	}
	shortTest = False # hardcode to True to run a quick debugging test
	if shortTest:
		print("+++++++++++++++++ WARNING: SHORT TEST +++++++++++++++++")
		hParams['datasetProportion'] = 0.0001
		hParams['numEpochs'] = 2

	if (expName is None):
		# Not running an experiment yet, so just return the "common" parameters
		return hParams

	dropProp = 0.0
	hParams['denseLayers'] = [int(x) for x in expName.split("_")]
	# hParams['resultsName'] = expName + "_results_freq"
	# hParams['predictName'] = expName + "_predict_freq"
	# hParams['percentName'] = expName + "_percent_freq"
	hParams['resultsName'] = expName + "_results"
	hParams['predictName'] = expName + "_predict"
	hParams['percentName'] = expName + "_percent"
	hParams['optimizer'] = 'adam'

	return hParams


def learn(dataSubsets, hParams):
	activities = ["pumping", "pushing", "coasting", "carving"]

	x_train, y_train, x_test, y_test, x_pred = dataSubsets

	x_train, y_train = correspondingShuffle(x_train, y_train)
	x_test, y_test = correspondingShuffle(x_test, y_test)

	print("y train is: ", y_train)	
	print("y test is: ", y_test)

	y_train = np.array([activities.index(x) for x in y_train])
	y_test = np.array([activities.index(x) for x in y_test])

	x_val = x_train[:int(hParams['valProportion'] * x_train.shape[0])]
	y_val = y_train[:int(hParams['valProportion'] * y_train.shape[0])]

	x_train = x_train[int(hParams['valProportion'] * x_train.shape[0]):]
	y_train = y_train[int(hParams['valProportion'] * y_train.shape[0]):]

	# == Sequential Constructor == #
	startTime = timeit.default_timer()
	model = tf.keras.Sequential()

	# long short term memory for time-series classification
	model.add(tf.keras.layers.LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
	model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
	model.add(tf.keras.layers.LSTM(128, activation='relu'))

	for layer in hParams['denseLayers']:
		model.add(tf.keras.layers.Dense(layer, activation='relu'))

	model.add(tf.keras.layers.Dense(4, activation='softmax'))

	# == Loss function == #
	lossFunc = tf.keras.losses.SparseCategoricalCrossentropy()

	# == fitting == #
	model.compile(loss=lossFunc, optimizer='adam', metrics=['accuracy', 'mae'])
	hist = model.fit(x_train, y_train, 
						validation_data=(x_val, y_val) 
							if hParams['valProportion']!=0.0 
							else None, 
			      		epochs=hParams['numEpochs'],
			      		verbose=1)	
	trainingTime = timeit.default_timer() - startTime
  
	# == Evaluation == #
	print('============ one unit 2 class, training set size:', x_train.shape[0], ' =============')
	print(model.summary())
	print('Training time:', trainingTime)
	print(model.evaluate(x_test, y_test))
	hParams['paramCount'] = model.count_params()
	pred = model.predict(x_test)
	predictions = []
	pos = ["pumping", "pushing", "coasting", "carving"]

	for i in range(len(pred)):
		m = pred[i].max()
		for j in range(len(pred[i])):
			if pred[i][j] == m:
				predictions.append(pos[j])

	print("y predict is: ", predictions)

	plotPredictions(y_test, predictions, hParams["experimentName"])

	return hist.history, model.evaluate(x_test, y_test), predictions

# chunk the position vectors into time window

models = [
	"10",
	"20_10",
	"50_10",
	"70_20",
	"100_50_10",
	"200_50_10",
	"300_50_10",
	"300_100_10",
	"300_200_10",
	"400_200_10",
	"300_200_100_10",
	"400_200_100_10",
	"450_400_300_200_10",
	"450_300_200_100_10"
]

main("LSTMF_")
# processResults()

buildOverallResults(models, "", "LSTMF")
