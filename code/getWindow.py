from dataTrans import *
from getFeatures import *
from scipy.signal import find_peaks
import scipy.stats
import scipy.io as sio
import matplotlib.pyplot as plt


# new idea
# Get chunks of 2-second of data
# get the angle 
# do the integration
# calculate the path vector (xx, xy, xz)
# apply the rotation matrix to turn it back to a default coordinate system
# apply the rotation matrix 

# spin it using pitch and azimuth 
# do this to get the direction that predominantly determines the pumping and pushing

# integrate acceleration data to get velocity and record the change of sign

# get a period, which means when it changes direction twice
def getTimes(y, timeData):
	peaks = find_peaks(y)[0]
	times = []
	for i in range(len(peaks)):
		times.append(timeData[peaks[i]])

	times.append(timeData[len(timeData) - 1])
	# print(times)

	# times = []
	# record = 0
	# index = 1

	# for i in range(1, len(y) - 1):
	# 	if (y[i - 1] < y[i] and y[i + 1] < y[i]) or (y[i - 1] > y[i] and y[i + 1] > y[i]):
	# 		if index % 2 != 0:
	# 			times.append(timeData[i] - record)	
	# 		else:
	# 			record = timeData[i]

	# 		index += 1

	return times


# get the features (12 features) for each window of half a period, including the one dominant frequency in that time period and the time of the window, since the larger the window the more it means that it is not a pumping period
def getFeaWindow(dataset):
	features = []
	j = 0
	dict = {"xx": [], "xy": [], "xz": [], "az": [], "ax": [], "ay": [], "Azimuth": [], "Pitch": [], "Roll": []}
	times = getTimes(dataset['x(t)(z)'], list(dataset['time']))
	time = []

	for i in range(len(dataset)):
		# windowx.append(dataset.iloc[i]['x(t)(x)'])
		# windowy.append(dataset.iloc[i]['x(t)(y)'])
		# windowz.append(dataset.iloc[i]['x(t)(z)'])

		xz = dataset.iloc[i]['x(t)(z)']
		xx = dataset.iloc[i]['x(t)(x)']
		xy = dataset.iloc[i]['x(t)(y)']
		az = dataset.iloc[i]['az']
		ax = dataset.iloc[i]['ax']
		ay = dataset.iloc[i]['ay']
		Azimuth = dataset.iloc[i]['Azimuth']
		Pitch = dataset.iloc[i]['Pitch']
		Roll = dataset.iloc[i]['Roll']

		dict['xx'].append(xx)
		dict['xy'].append(xy)
		dict['xz'].append(xz)
		dict['ax'].append(ax)
		dict['ay'].append(ay)
		dict['az'].append(az)
		dict['Azimuth'].append(Azimuth)
		dict['Pitch'].append(Pitch)
		dict['Roll'].append(Roll)
		time.append(dataset.iloc[i]['time'])

		if j < len(times) and i < len(dataset) and dataset.iloc[i].time == times[j]: # -> it should be the sum so far
			# freqx, freqy, freqz = fft(np.array(freqx)), fft(np.array(freqy)), fft(np.array(freqz))
			plt.plot(np.array(time), dict['xz'])
			feature = getFeatures(dict, times[j])
			features.append(feature)
			# windowx, windowy, windowz = [], [], []
			dict = {"xx": [], "xy": [], "xz": [], "az": [], "ax": [], "ay": [], "Azimuth": [], "Pitch": [], "Roll": []}
			# cur += times[j]
			j += 1
			time = []

		# cur = 0
		# if i < len(dataset) and dataset.iloc[i].time > 0 and dataset.iloc[i].time >= cur: # -> it should be the sum so far
		# 	# freqx, freqy, freqz = fft(np.array(freqx)), fft(np.array(freqy)), fft(np.array(freqz))
		# 	print(cur)
		# 	plt.plot(np.array(time), dict['xz'])

		# 	# feature = getFeatures(dict, times[j])
		# 	feature = getFeatures(dict, 2)
		# 	features.append(feature)
		# 	# windowx, windowy, windowz = [], [], []
		# 	dict = {"xx": [], "xy": [], "xz": [], "az": [], "ax": [], "ay": [], "Azimuth": [], "Pitch": [], "Roll": []}
		# 	cur += 1
		# 	# j += 1
		# 	time = []

	plt.savefig('../graphs/period')
	print("shape of the features array is", np.array(features).shape)

	return np.array(features) 

def getFeaWindowConstSec(dataset, numSec):
	features = []
	j = 0
	dict = {"xx": [], "xy": [], "xz": [], "az": [], "ax": [], "ay": [], "Azimuth": [], "Pitch": [], "Roll": []}
	xangss, yangss, zangss = [], [], []
	last = []
	xLast, yLast, zLast = [], [], []

	for i in range(len(dataset)):
		xz = dataset.iloc[i]['x(t)(z)']
		xx = dataset.iloc[i]['x(t)(x)']
		xy = dataset.iloc[i]['x(t)(y)']
		az = dataset.iloc[i]['az']
		ax = dataset.iloc[i]['ax']
		ay = dataset.iloc[i]['ay']
		Azimuth = dataset.iloc[i]['Azimuth']
		Pitch = dataset.iloc[i]['Pitch']
		Roll = dataset.iloc[i]['Roll']

		dict['xx'].append(xx)
		dict['xy'].append(xy)
		dict['xz'].append(xz)
		dict['ax'].append(ax)
		dict['ay'].append(ay)
		dict['az'].append(az)
		dict['Azimuth'].append(Azimuth)
		dict['Pitch'].append(Pitch)
		dict['Roll'].append(Roll)
		# time.append(dataset.iloc[i]['time'])

		if i == len(dataset) - 1 or (i < len(dataset) and dataset.iloc[i].time - j * numSec >= numSec): # -> it should be the sum so far
			print(dataset.iloc[i].time, j * numSec)
			newRoll, newAzimuth = normalize_vector(dict['xx'], dict['xy'], dict['xz'])
			# xangss += xangs 
			# yangss += yangs 
			# zangss += zangs
			for i in range(len(dict['xx'])):
				matrix = (dict['xx'][i], dict['xy'][i], dict['xz'][i])
				last = rotation(matrix, newAzimuth, 0, newRoll)
				dict['xx'][i] = last[0]
				dict['xy'][i] = last[1]
				dict['xz'][i] = last[2]
 
			feature = getConstFeatures(dict)
			features.append(feature)
			dict = {"xx": [], "xy": [], "xz": [], "az": [], "ax": [], "ay": [], "Azimuth": [], "Pitch": [], "Roll": []}
			j += 1

	# for i in range(len(dataset)):
	# 	matrix = [dataset['x(t)(x)'].iat[i], dataset['x(t)(y)'].iat[i], dataset['x(t)(z)'].iat[i]]
	# 	last = rotation(matrix, zangss[i], 0, xangss[i])
	# 	xLast.append(last[0])
	# 	yLast.append(last[1])
	# 	zLast.append(last[2])

	# dataset['xLast'], dataset['yLast'], dataset['zLast'] = xLast, yLast, zLast 

	return np.array(features), dataset


