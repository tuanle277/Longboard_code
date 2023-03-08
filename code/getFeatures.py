import numpy as np
from scipy.fft import rfft, rfftfreq


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

def getFeatures(dict, time):
	features = []
	dataLength = len(dict['ax'])

	devA = abs(dict['Azimuth'][-1] - dict['Azimuth'][0]) / time
	devP = abs(dict['Pitch'][-1] - dict['Pitch'][0]) / time
	devR = abs(dict['Roll'][-1] - dict['Roll'][0]) / time

	least = min(devR, devP, devA)
	most = max(devR, devP, devA)

	if least == devP: 
		a = np.array(dict["ay"])
		x1 = np.array(dict["xx"])
		x2 = np.array(dict["xz"])

	elif least == devR: 
		a = np.array(dict["ax"])
		x1 = np.array(dict["xz"])
		x2 = np.array(dict["xy"])

	else:
		a = np.array(dict["az"])
		x1 = np.array(dict["xx"])
		x2 = np.array(dict["xy"])

	if most == devP:
		y = np.array(dict["xy"])
	elif most == devR:
		y = np.array(dict["xx"])
	else:
		y = np.array(dict["xz"])


	newthing1 = np.array([x - x1.mean() for x in x1])
	newthing2 = np.array([x - x2.mean() for x in x2])

	yf1 = np.abs(rfft(newthing1))
	yf2 = np.abs(rfft(newthing2))

	xf1 = rfftfreq(len(x1), 1 / len(x1))
	xf2 = rfftfreq(len(x2), 1 / len(x2))

	# plt.plot(xf1[:20] + xf2[:20], yf1[:20])
	# plt.savefig("graphs/fourier")

	peak1 = yf1.max()
	peak2 = yf2.max()

	freq1 = 0
	freq2 = 0

	for i in range(len(xf1)):
		if yf1[i] == peak1:
			freq1 = xf1[i]

	for i in range(len(xf2)):
		if yf2[i] == peak2:
			freq2 = xf2[i]

	n5 = np.nanpercentile(y, 5)
	n25 = np.nanpercentile(y, 25)
	n75 = np.nanpercentile(y, 75)
	n95 = np.nanpercentile(y, 95)
	median = np.nanpercentile(y, 50)
	mean = np.nanmean(y)
	std = np.nanstd(y)
	var = np.nanvar(y)
	rms = np.nanmean(np.sqrt(y**2))

	features.append(freq1 + freq2)
	features.append(y.max())
	features.append(y.min())
	features.append(n5)
	features.append(n25)
	features.append(n75)
	features.append(n95)
	features.append(median)
	features.append(mean)
	features.append(std)
	features.append(var)
	features.append(rms)
	# features.append(time)
	features.append(np.nanmean(a))

	return features

def getConstFeatures(dict):
	features = []
	dataLength = len(dict['ax'])

	a = np.array(dict["ay"])
	x1 = np.array(dict["xx"])
	x2 = np.array(dict["xz"])

	newthing1 = np.array([x - x1.mean() for x in x1])
	newthing2 = np.array([x - x2.mean() for x in x2])

	yf1 = np.abs(rfft(newthing1))
	yf2 = np.abs(rfft(newthing2))

	xf1 = rfftfreq(len(x1), 1 / len(x1))
	xf2 = rfftfreq(len(x2), 1 / len(x2))

	y = x1

	# plt.plot(xf1[:20] + xf2[:20], yf1[:20])
	# plt.savefig("graphs/fourier")

	peak1 = yf1.max()
	peak2 = yf2.max()

	freq1 = 0
	freq2 = 0

	for i in range(len(xf1)):
		if yf1[i] == peak1:
			freq1 = xf1[i]

	for i in range(len(xf2)):
		if yf2[i] == peak2:
			freq2 = xf2[i]

	n5 = np.nanpercentile(y, 5)
	n25 = np.nanpercentile(y, 25)
	n75 = np.nanpercentile(y, 75)
	n95 = np.nanpercentile(y, 95)
	median = np.nanpercentile(y, 50)
	mean = np.nanmean(y)
	std = np.nanstd(y)
	var = np.nanvar(y)
	rms = np.nanmean(np.sqrt(y**2))

	features.append(freq1 + freq2)
	features.append(y.max())
	features.append(y.min())
	features.append(n5)
	features.append(n25)
	features.append(n75)
	features.append(n95)
	features.append(median)
	features.append(mean)
	features.append(std)
	features.append(var)
	features.append(rms)
	# features.append(time)
	features.append(np.nanmean(a))

	return features