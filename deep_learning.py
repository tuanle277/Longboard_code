import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
import timeit

import matplotlib.pyplot as plt
import pywt
import scipy.stats
import json

import datetime as dt

from collections import defaultdict, Counter
from scipy.fft import fft, fftfreq


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



def main():
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
		"100_50_10",
		"200_50_10",
		"300_50_10",
		"300_100_10",
		"300_200_10",
		"400_200_10",
		"300_200_100_10",
		"400_200_100_10",
		"500_400_300_200_10",
		"500_300_200_100_10"
	]


	x_train, y_train = getFrequenciesTrainData()
	# x_test, y_test = getTestData()
	x_test, y_test = x_train[:int(0.2 * x_train.shape[0])], y_train[:int(0.2 * y_train.shape[0])]
	x_pred = getFrequenciesPredData()

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


# different data normalization methods were tested 
def normalized_data(fileName):
	columns = ["az", "ay", "ax", "time", "Azimuth", "Pitch", "Roll"]

	df = pd.read_csv(fileName, usecols=columns)

	meanx, stdx = df.ax.mean(), df.ax.std()
	meany, stdy = df.ay.mean(), df.ay.std()
	meanz, stdz = df.az.mean(), df.az.std()

	maxx, minx = df.ax.max(), df.ax.min()
	maxy, miny = df.ay.max(), df.ay.min()	
	maxz, minz = df.az.max(), df.az.min()


	# for i in range(len(df)):
	# 	df.ax[i] = (df.ax[i] - meanx) / stdx
	# 	df.ay[i] = (df.ay[i] - meany) / stdy
	# 	df.az[i] = (df.az[i] - meanz) / stdz

	for i in range(len(df)):
		df.ax[i] = (df.ax[i] - minx) / (maxx - minx)
		df.ay[i] = (df.ay[i] - miny) / (maxy - miny)
		df.az[i] = (df.az[i] - minz) / (maxz - minz)

	return df


def getPreqWindow(dataset, time):
	frequencies = []
	windowx, windowy, windowz = [], [], []
	cur_time = 0
	for i in range(len(dataset)):
		windowx.append(dataset.iloc[i]['x(t)(x)'])
		windowy.append(dataset.iloc[i]['x(t)(y)'])
		windowz.append(dataset.iloc[i]['x(t)(z)'])
		if i < len(dataset) and cur_time >= time: 
			freqx, freqy, freqz = fft(np.array(freqx)), fft(np.array(freqy)), fft(np.array(freqz))
			frequencies.append((freqx, freqy, freqz))
			windowx, windowy, windowz = [], [], []
			cur_time = 0

		cur_time += dataset.iloc[i].time

	return np.array(frequencies)


# =============== get frequencies after seconds in dataset ============== #
def getFrequenciesTrainData():
	fileNames = ["pumping_.csv", "pushing_.csv", "pushing_2.csv", "coasting_.csv"]

	# df_train = pd.DataFrame(columns = ['time', 'ax', 'ay', 'az'])
	# time = 0
	# for fileName in fileNames:
	# 	df = pd.read_csv(fileName, usecols=columns)
	# 	if fileName == "pumping.csv":
	# 		df = df.loc[(df["time"] > 10) & (df["time"] < df.iloc[-1]["time"] - 3)]
	# 		for i in range(len(df)):
	# 			time += df.iloc[i].time
	# 			df_train = df_train.append({'time': time, 'ax': df.iloc[i].ax, 'ay': df.iloc[i].ay, 'az': df.iloc[i].az}, ignore_index=True)

	# df_train = transformation(df_train)
	# x_train = getPreqWindow(df_train, time)	

	# return frequencies

	x_train = np.array([[0, 0, 0]])
	y_train = [""]
	for fileName in fileNames:
		xx, xy, xz = [], [], []
		df = normalized_data(fileName)

		if fileName == "pumping.csv":
			df = df.loc[(df["time"] > 10) & (df["time"] < df.iloc[-1]["time"] - 3)]

		df = transformation(df)
		print(len(df))
		for i in range(len(df)):
			xx.append(df.iloc[i]["x(t)(x)"])
			xy.append(df.iloc[i]["x(t)(y)"])
			xz.append(df.iloc[i]["x(t)(z)"])

		freqx, freqy, freqz = fft(np.array(xx)), fft(np.array(xy)), fft(np.array(xz))
		x_train = np.concatenate((x_train, np.concatenate((freqx, freqy, freqz), axis=0).reshape((freqx.shape[0], 3))))

		y_train += [fileName.split("_")[0]] * freqx.shape[0]

	x_train = np.array(x_train)

	for i in range(len(x_train)):
		if (x_train[i] == [0, 0, 0]).any():
			x_train = np.delete(x_train, i, 0)
			y_train = np.delete(y_train, i, 0)
		if i == len(x_train) - 1:
			break

	y_train = np.array(y_train)
	return x_train, y_train

def getFrequenciesPredData():
	fileNames = ["longboard.csv", "longboard2.csv", "mixed.csv", "mixed (pushing, pumping, coasting. carving).csv"]
	columns = ["az", "ay", "ax", "time", "Azimuth", "Pitch", "Roll"]
	x_pred = np.array([[0, 0, 0]])
	for fileName in fileNames:
		xx, xy, xz = [], [], []
		df = normalized_data(fileName)

		df = transformation(df)
		for i in range(len(df)):
			xx.append(df.iloc[i]["x(t)(x)"])
			xy.append(df.iloc[i]["x(t)(y)"])
			xz.append(df.iloc[i]["x(t)(z)"])

		freqx, freqy, freqz = fft(np.array(xx)), fft(np.array(xy)), fft(np.array(xz))
		x_pred = np.concatenate((x_pred, np.concatenate((freqx, freqy, freqz), axis=0).reshape((freqx.shape[0], 3))))

	for i in range(len(x_pred)):
		if (x_pred[i] == [0, 0, 0]).any():
			x_pred = np.delete(x_pred, i, 0)
		if i == len(x_pred) - 1:
			break

	return x_pred

# def getFrequenciesTestData(t = 0.36):
# 	fileNames = ["longboard.csv", "longboard2.csv", "mixed.csv", "mixed (pushing, pumping, coasting. carving).csv"]
# 	columns = ["az", "ay", "ax", "time"]


# 	frequencies = []
# 	acx, acy, acz = [], [], []
# 	time = 0 
# 	for fileName in fileNames:
# 		df = pd.read_csv(fileName, usecols=columns)

# 		for i in range(len(df)):
# 			time += df.iloc[i].time 
# 			acx.append(df.iloc[i].ax)
# 			acy.append(df.iloc[i].ay)
# 			acz.append(df.iloc[i].az)
# 			if i < len(df) and time >= t:
# 				freqx, freqy, freqz = fft(np.array(acx)), fft(np.array(acy)), fft(np.array(acz))
# 				frequencies.append((freqx, freqy, freqz))
# 				acx, acy, acz = [], [], []
# 				time = 0

# 	x_train, y_train = np.array(frequencies), np.array(["pumping"])
# 	return x_train, y_train
	
def getTrainData():
	fileNames = ["pumping_.csv", "pushing_.csv", "pushing_2.csv", "coasting_.csv"]

	# df.loc[~(df==0).all(axis=1)]
	columns = ["az", "ay", "ax", "time", "Azimuth", "Pitch", "Roll"]
	x_train, y_train = np.array([[0, 0, 0]]), np.array(["Skip"])
	for fileName in fileNames:
		df = normalized_data(fileName)

		if fileName.split('_')[0] == "pumping":
			df = df.loc[(df["time"] > 10) & (df["time"] < df.iloc[-1]["time"] - 3)]
			df.duplicated(keep='first')
			y_train = np.append(y_train, ["pumping"] * len(df))

		elif fileName.split('_')[0] == "pushing":
			y_train = np.append(y_train, ["pushing"] * len(df))

		else: 
			y_train = np.append(y_train, ["coasting"] * len(df))

		for i in range(len(df)):
			x_train = np.append(x_train, [[df.iloc[i].ax, df.iloc[i].ay, df.iloc[i].az]], axis=0)

	for i in range(len(x_train)):
		if (x_train[i] == [0, 0, 0]).any():
			x_train = np.delete(x_train, i, 0)
			y_train = np.delete(y_train, i, 0)
		if i == len(x_train) - 1:
			break

	return x_train, y_train

def getTestData():
	x_test, y_test = np.array([[0, 0, 0]]), np.array(["Skip"])
	fileNames = ["longboard.csv", "longboard2.csv", "mixed.csv", "mixed (pushing, pumping, coasting. carving).csv"]
	columns = ["az", "ay", "ax", "time"]
	for fileName in fileNames:
		df = normalized_data(fileName)
		y_test = np.append(y_test, ["pumping"] * len(df))
		for i in range(len(df)):
	  		x_test = np.append(x_test, [[df.iloc[i].ax, df.iloc[i].ay, df.iloc[i].az]], axis=0)

		for i in range(len(x_test)):
			if (x_test[i] == [0, 0, 0]).any():
			  	x_test = np.delete(x_test, i, 0)
			  	y_test = np.delete(y_test, i, 0)
			if i == len(x_test) - 1:
	  			break

	return x_test, y_test

def getPredData():
	x_pred = np.array([[0, 0, 0]])
	fileNames = ["longboard.csv", "longboard2.csv", "mixed.csv", "mixed (pushing, pumping, coasting. carving).csv"]
	columns = ["az", "ay", "ax", "time"]
	for fileName in fileNames:
		df = normalized_data(fileName)
		for i in range(len(df)):
			x_pred = np.append(x_pred, [[df.iloc[i].ax, df.iloc[i].ay, df.iloc[i].az]], axis=0)

	for i in range(len(x_pred)):
		if (x_pred[i] == [0, 0, 0]).any():
			x_pred = np.delete(x_pred, i, 0)
		if i == len(x_pred) - 1:
			break

	return x_pred

def calculate_entropy(list_values):
	counter_values = Counter(list_values).most_common()
	probabilities = [elem[1]/len(list_values) for elem in counter_values]
	entropy=scipy.stats.entropy(probabilities)
	return entropy

def calculate_statistics(list_values):
	n5 = np.nanpercentile(list_values, 5)
	n25 = np.nanpercentile(list_values, 25)
	n75 = np.nanpercentile(list_values, 75)
	n95 = np.nanpercentile(list_values, 95)
	median = np.nanpercentile(list_values, 50)
	mean = np.nanmean(list_values)
	std = np.nanstd(list_values)
	var = np.nanvar(list_values)
	rms = np.nanmean(np.sqrt(list_values**2))
	return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
	zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
	no_zero_crossings = len(zero_crossing_indices)
	mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
	no_mean_crossings = len(mean_crossing_indices)
	return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
	entropy = calculate_entropy(list_values)
	crossings = calculate_crossings(list_values)
	statistics = calculate_statistics(list_values)
	return [entropy] + crossings + statistics

def feature_extraction(dataset, waveletname):
	a = []
	for signal_no in range(0, len(dataset)):
		features = []
		signal = dataset[signal_no]
		# print(signal)
		list_coeff = pywt.wavedec(signal, waveletname)
		for coeff in list_coeff:
			features += get_features(coeff)
		a.append(features)
	# X = np.array(features)
	return a

def correspondingShuffle(x, y):
	indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
	shuffled_indices = tf.random.shuffle(indices)

	shuffled_x = tf.gather(x, shuffled_indices)
	shuffled_y = tf.gather(y, shuffled_indices)

	return shuffled_x, shuffled_y


def writeExperimentalResults(hParams, trainResults, testResults, predictions):
	f = open("results/" + hParams["resultsName"] + ".txt", 'w')
	f.write(str(hParams) + '\n\n')
	f.write(str(trainResults) + '\n\n')
	f.write(str(testResults) + '\n\n')

	f1 = open("results/" + hParams["predictName"] + ".txt", 'w')
	f1.write(str(hParams) + '\n\n')
	f1.write(str(predictions))

	percentReport = prediction_result(predictions)
	writePercentReport(hParams, percentReport)

	f.close()
	f1.close()

def writePercentReport(hParams, percentReport):
	ff = open("results/" + hParams["percentName"] + ".txt", 'w')
	ff.write(percentReport)
	ff.close()

def writePredictions(hParams, predictions):
	ff = open("results/" + hParams["predictName"] + ".txt", 'w')
	ff.write(str(predictions))
	ff.close()

def readExperimentalResults(fileName):
	f = open("results/" + fileName + ".txt",'r')
	data = f.read().split('\n\n')

	data[0] = data[0].replace("\'", "\"")
	data[1] = data[1].replace("\'", "\"")

	hParams = json.loads(data[0])
	trainResults = json.loads(data[1])
	testResults = json.loads(data[2])

	return hParams, trainResults, testResults

def readPredResults(fileName):
	f = open("results/" + fileName + ".txt", "r")
	data = f.read().split('\n\n')

	data[0] = data[0].replace("\'", "\"")
	data[1] = data[1].replace("\'", "\"")

	hParams = json.loads(data[0])
	predictions = json.loads(data[1])
	return hParams, predictions

def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
	fig, ax = plt.subplots()
	y = np.array(yList).transpose()
	ax.plot(x, y)
	ax.set(xlabel=xLabel, title=title)
	plt.legend(yLabelList, loc='best', shadow=True)
	ax.grid()
	yLabelStr = "__" + "__".join([label for label in yLabelList])
	filepath = "results/" + title + " " + yLabelStr + ".png"
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
	filepath = "results/" + filename + ".png"
	plt.savefig(filepath)
	print("Figure saved in", filepath)

def processResults():
	hParams, trainResults, testResults = readExperimentalResults("freq_test_results")
	hParams, dPredictions = readPredResults("freq_predictions")

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

def buildValAccuracyPlot(fileNames, title):
    # == get hParams == #
    hParams = readExperimentalResults(fileNames[0]+"_results_freq")[0]

    # == plot curves with yList being the validation accuracies == #
    plotCurves(x = np.arange(0, hParams["numEpochs"]), 
            yList=[readExperimentalResults(name+"_results_freq")[1]['val_accuracy'] for name in fileNames], 
            xLabel="Epoch",
            yLabelList=fileNames,
            title= "val_" + title)

    plotCurves(x = np.arange(0, hParams["numEpochs"]), 
            yList=[readExperimentalResults(name+"_results_freq")[1]['accuracy'] for name in fileNames], 
            xLabel="Epoch",
            yLabelList=fileNames,
            title= "acc_" + title)

    # == plot points with xList being the parameter counts of all and yList being the test accuracies == #
    plotPoints(xList=[readExperimentalResults(name+"_results_freq")[0]['paramCount'] for name in fileNames],
                yList=[readExperimentalResults(name+"_results_freq")[2][0] for name in fileNames],
                pointLabels= [name for name in fileNames],
                xLabel='Number of parameters',
                yLabel='Test set loss',
                title="Test set loss_" + title,
                filename="Test set loss_" + title)



def prediction_result(predictions):
	pred = {"pumping": 0, "pushing": 0, "coasting": 0}
	for p in predictions:
		pred[p] += 1

	percentReport = "Percentage of pumping: " + "{:.1f}%".format(100 * pred["pumping"]/len(predictions), "%") + '\n' + "Percentage of pushing: " + "{:.1f}%".format(100 * pred["pushing"]/len(predictions), "%")+ '\n' + "Percentage of coasting: " + "{:.1f}%".format(100 * pred["coasting"]/len(predictions), "%")
	return percentReport


def rotation(matrix, alpha, beta, theta):
	alpha, beta, theta = -alpha * np.pi / 180, -beta * np.pi / 180, -theta * np.pi / 180
	rotation_matrix = [[np.cos(alpha) * np.cos(beta), np.cos(alpha) * np.sin(beta) * np.sin(theta) - np.sin(alpha) * np.cos(theta), np.cos(alpha) * np.sin(beta) * np.cos(theta) + np.sin(alpha) * np.sin(theta)]
	                 ,[np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta) * np.sin(theta) + np.cos(alpha) * np.cos(theta), np.sin(alpha) * np.sin(beta) * np.sin(theta) - np.cos(alpha) * np.sin(theta)]
	                 ,[-np.sin(beta), np.cos(beta) * np.sin(theta), np.cos(beta) * np.cos(theta)]]


	return np.matmul(matrix, rotation_matrix) 

def transformation(df):

	transformed_acceleration = []

	for index in range(len(df)):
		alpha, beta, theta = df['Azimuth'].iat[index], df['Pitch'].iat[index], df['Roll'].iat[index]
		acceleration_matrix = [df['ax'].iat[index], df['ay'].iat[index], df['az'].iat[index]]
		transformed_acceleration.append(rotation(acceleration_matrix, alpha, beta, theta))

	df['transformed_ax'] = [x[0] for x in transformed_acceleration]
	df['transformed_ay'] = [x[1] for x in transformed_acceleration]
	df['transformed_az'] = [x[2] for x in transformed_acceleration]

	delta_v, xtraw, delta_t, prev_pos, pos = 0, [[0, 0, 0]], 0, [0] * 3, [0] * 3

	for i in range(len(df) - 1):
		delta_t = df.iloc[i + 1]['time'] - df.iloc[i]['time']
		delta_v = [delta_t * df.iloc[i]['ax'], delta_t * df.iloc[i]['ay'], delta_t * df.iloc[i]['az']]
		pos = [x + y for (x, y) in zip([2 * y - z for (y, z) in zip(pos, prev_pos)], [delta_t * delta_v[0], delta_t * delta_v[1], delta_t * delta_v[2]])]
		prev_pos = pos
		xtraw.append(pos)

	df['x(t)(x)raw'] = [x[0] for x in xtraw] 
	df['x(t)(y)raw'] = [x[1] for x in xtraw]
	df['x(t)(z)raw'] = [x[2] for x in xtraw]

	delta_v, xt, delta_t, prev_pos, pos = 0, [[0, 0, 0]], 0, [0] * 3, [0] * 3

	for i in range(len(df) - 1):
		delta_t = df.iloc[i + 1]['time'] - df.iloc[i]['time']
		delta_v = [delta_t * df.iloc[i]['transformed_ax'], delta_t * df.iloc[i]['transformed_ay'], delta_t * df.iloc[i]['transformed_az']]
		pos = [x + y for (x, y) in zip([2 * y - z for (y, z) in zip(pos, prev_pos)], [delta_t * delta_v[0], delta_t * delta_v[1], delta_t * delta_v[2]])]
		prev_pos = pos
		xt.append(pos)

	df['x(t)(x)'] = [x[0] for x in xt] 
	df['x(t)(y)'] = [x[1] for x in xt]
	df['x(t)(z)'] = [x[2] for x in xt]

	return df

def getHParams(expName=None):
	# Set up what's the same for each experiment
	hParams = {
		'experimentName': expName,
		'datasetProportion': 1.0,
		'valProportion': 0.1,
		'numEpochs': 20
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
	hParams['resultsName'] = expName + "_results_freq"
	hParams['predictName'] = expName + "_predict_freq"
	hParams['percentName'] = expName + "_percent_freq"
	hParams['optimizer'] = 'adam'

	return hParams


def learn(dataSubsets, hParams):
	x_train, y_train, x_test, y_test, x_pred = dataSubsets

	x_train, y_train = correspondingShuffle(x_train, y_train)
	x_test, y_test = correspondingShuffle(x_test, y_test)

	y_train = np.array([int(x == "pumping") for x in y_train])
	y_test = np.array([int(x == "pumping") for x in y_test])

	x_val = x_train[:int(hParams['valProportion'] * x_train.shape[0])]
	y_val = y_train[:int(hParams['valProportion'] * y_train.shape[0])]

	x_train = x_train[int(hParams['valProportion'] * x_train.shape[0]):]
	y_train = y_train[int(hParams['valProportion'] * y_train.shape[0]):]

	print(x_train.shape)

	# == Sequential Constructor == #
	startTime = timeit.default_timer()
	model = tf.keras.Sequential()

	for layer in hParams['denseLayers']:
		model.add(tf.keras.layers.Dense(layer, activation='relu'))

	model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

	# == Loss function == #
	lossFunc = tf.keras.losses.SparseCategoricalCrossentropy()

	# == fitting == #
	model.compile(loss=lossFunc, optimizer='adam', metrics=['accuracy'])
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
	pred = model.predict(x_pred)
	predictions = []
	pos = ["pumping", "pushing", "coasting"]
	for i in range(len(pred)):
		m = pred[i].max()
		for j in range(len(pred[i])):
			if pred[i][j] == m:
				predictions.append(pos[j])

	return hist.history, model.evaluate(x_test, y_test), predictions



models = [
	"100_50_10",
	"200_50_10",
	"300_50_10",
	"300_100_10",
	"300_200_10",
	"400_200_10",
	"300_200_100_10",
	"400_200_100_10",
	"500_400_300_200_10",
	"500_300_200_100_10"
]

main()
# processResults()

buildValAccuracyPlot(models, "dense")
