def writeExperimentalResults(hParams, trainResults, testResults, predictions):
	f = open("results/models/LSTM/" + hParams["resultsName"] + "_LSTM" + ".txt", 'w')
	f.write(str(hParams) + '\n\n')
	f.write(str(trainResults) + '\n\n')
	f.write(str(testResults) + '\n\n')

	f1 = open("results/models/LSTM/" + hParams["predictName"] + "_LSTM" + ".txt", 'w')
	f1.write(str(hParams) + '\n\n')
	f1.write(str(predictions))

	percentReport = prediction_result(predictions)
	writePercentReport(hParams, percentReport)

	f.close()
	f1.close()

def writePercentReport(hParams, percentReport):
	ff = open("results/models/LSTM/" + hParams["percentName"] + "_LSTM" + ".txt", 'w')
	ff.write(percentReport)
	ff.close()

def writePredictions(hParams, predictions):
	ff = open("results/models/LSTM/" + hParams["predictName"] + "_LSTM" + ".txt", 'w')
	ff.write(str(predictions))
	ff.close()

def readExperimentalResults(fileName):
	f = open("results/models/LSTM/" + fileName + ".txt",'r')
	data = f.read().split('\n\n')

	data[0] = data[0].replace("\'", "\"")
	data[1] = data[1].replace("\'", "\"")

	hParams = json.loads(data[0])
	trainResults = json.loads(data[1])
	testResults = json.loads(data[2])

	return hParams, trainResults, testResults

def readPredResults(fileName):
	f = open("results/models/LSTM/" + fileName + ".txt", "r")
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
	filepath = "results/models/LSTM/" + title + " " + yLabelStr + ".png"
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
	filepath = "results/models/LSTM/" + filename + ".png"
	plt.savefig(filepath)
	print("Figure saved in", filepath)

===========================================================================================================================
===========================================================================================================================

def writeExperimentalResults(hParams, trainResults, testResults, predictions):
	f = open("results/models/features/" + hParams["resultsName"] + "_features" + ".txt", 'w')
	f.write(str(hParams) + '\n\n')
	f.write(str(trainResults) + '\n\n')
	f.write(str(testResults) + '\n\n')

	f1 = open("results/models/features/" + hParams["predictName"] + "_features" + ".txt", 'w')
	f1.write(str(hParams) + '\n\n')
	f1.write(str(predictions))

	percentReport = prediction_result(predictions)
	writePercentReport(hParams, percentReport)

	f.close()
	f1.close()

def writePercentReport(hParams, percentReport):
	ff = open("results/models/features/" + hParams["percentName"] + ".txt", 'w')
	ff.write(percentReport)
	ff.close()

def writePredictions(hParams, predictions):
	ff = open("results/models/features/" + hParams["predictName"] + ".txt", 'w')
	ff.write(str(predictions))
	ff.close()

def readExperimentalResults(fileName):
	f = open("results/models/features/" + fileName + ".txt",'r')
	data = f.read().split('\n\n')

	data[0] = data[0].replace("\'", "\"")
	data[1] = data[1].replace("\'", "\"")

	hParams = json.loads(data[0])
	trainResults = json.loads(data[1])
	testResults = json.loads(data[2])

	return hParams, trainResults, testResults

def readPredResults(fileName):
	f = open("results/models/features/" + fileName + ".txt", "r")
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
	filepath = "results/models/features/" + title + " " + yLabelStr + ".png"
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
	filepath = "results/models/features/" + filename + ".png"
	plt.savefig(filepath)
	print("Figure saved in", filepath)

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