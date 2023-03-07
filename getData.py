# =============== get frequencies after dynamic seconds in dataset ============== #
def getFeaturesTrainData():
	fileNames = ["pumping_.csv", "pushing_.csv", "pushing_2.csv", "coasting_.csv"]

	x_train = np.array([[0] * 13])
	y_train = [""]
	for fileName in fileNames:
		df = normalized_data(fileName)

		if fileName == "pumping.csv":
			df = df.loc[(df["time"] > 10) & (df["time"] < df.iloc[-1]["time"] - 3)]

		df = transformation(df)
		features = getFeaWindow(df)
		x_train = np.concatenate((x_train, features))

		y_train += [fileName.split("_")[0]] * features.shape[0]

	for i in range(len(x_train)):
		if (x_train[i] == [0]*13).any():
			x_train = np.delete(x_train, i, 0)
			y_train = np.delete(y_train, i, 0)

		if i >= len(x_train) - 1:
			break

	print(x_train.shape, y_train.shape)

	y_train = np.array(y_train)
	return x_train, y_train

def getFeaturesPredData():
	fileNames = ["longboard.csv", "longboard2.csv", "mixed.csv", "mixed (pushing, pumping, coasting. carving).csv"]

	x_pred = np.array([[0] * 13])
	for fileName in fileNames:
		df = normalized_data(fileName)

		df = transformation(df)
		features = getFeaWindow(df)
		x_pred = np.concatenate((x_pred, features))

	for i in range(len(x_pred)):
		if (x_pred[i] == [0]*13).any():
			x_pred = np.delete(x_pred, i, 0)
		if i >= len(x_pred) - 1:
			break

	return x_pred

# =============== get frequencies after constant seconds in dataset ============== #
def getConFeaturesTrainData():
	fileNames = ["pumping_.csv", "pushing_.csv", "pushing_2.csv", "coasting_.csv"]

	x_train = np.array([[0] * 13])
	y_train = [""]
	for fileName in fileNames:
		df = normalized_data(fileName)

		if fileName == "pumping.csv":
			df = df.loc[(df["time"] > 10) & (df["time"] < df.iloc[-1]["time"] - 3)]

		df = transformation(df)
		features = getFeaWindowConstSec(df, 2)
		x_train = np.concatenate((x_train, features))

		y_train += [fileName.split("_")[0]] * features.shape[0]

	for i in range(len(x_train)):
		if (x_train[i] == [0]*13).any():
			x_train = np.delete(x_train, i, 0)
			y_train = np.delete(y_train, i, 0)

		if i >= len(x_train) - 1:
			break

	print(x_train.shape, y_train.shape)

	y_train = np.array(y_train)
	return x_train, y_train

def getConFeaturesPredData():
	fileNames = ["longboard.csv", "longboard2.csv", "mixed.csv", "mixed (pushing, pumping, coasting. carving).csv"]

	x_pred = np.array([[0] * 13])
	for fileName in fileNames:
		# xx, xy, xz = [], [], []
		df = normalized_data(fileName)

		df = transformation(df)
		features = getFeaWindow2Sec(df, 2)
		x_pred = np.concatenate((x_pred, features))

	for i in range(len(x_pred)):
		if (x_pred[i] == [0]*13).any():
			x_pred = np.delete(x_pred, i, 0)
		if i >= len(x_pred) - 1:
			break

	return x_pred

# =============== get displacements after seconds in dataset ============== #
def getPathTrainData():
	fileNames = ["pumping_.csv", "pushing_.csv", "pushing_2.csv", "coasting_.csv"]

	x_train = np.array([[0] * 3])
	y_train = [""]
	for fileName in fileNames:
		# xx, xy, xz = [], [], []
		df = normalized_data(fileName)

		if fileName == "pumping.csv":
			df = df.loc[(df["time"] > 10) & (df["time"] < df.iloc[-1]["time"] - 3)]

		df = transformation(df)
		features = getPathWindow(df)
		x_train = np.concatenate((x_train, features))

		y_train += [fileName.split("_")[0]] * features.shape[0]

	for i in range(len(x_train)):
		if (x_train[i] == [0]*3).any():
			x_train = np.delete(x_train, i, 0)
			y_train = np.delete(y_train, i, 0)

		if i >= len(x_train) - 1:
			break

	y_train = np.array(y_train)
	return x_train, y_train

def getPathPredData():
	fileNames = ["longboard.csv", "longboard2.csv", "mixed.csv", "mixed (pushing, pumping, coasting. carving).csv"]

	x_pred = np.array([[0] * 3])
	for fileName in fileNames:
		df = normalized_data(fileName)

		df = transformation(df)

		features = getPathWindow(df)
		x_pred = np.concatenate((x_pred, features))

	for i in range(len(x_pred)):
		if (x_pred[i] == [0]*3).any():
			x_pred = np.delete(x_pred, i, 0)
		if i >= len(x_pred) - 1:
			break

	return x_pred
