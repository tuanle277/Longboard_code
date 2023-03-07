def rotation(matrix, alpha, beta, theta):
	print(alpha, beta, theta)
	if alpha > 180:
		alpha = - (180 - alpha % 180)

	if beta > 180:
		beta = - (180 - beta % 180)

	if theta > 180:
		theta = - (180 - theta % 180)

	print(alpha, beta, theta)
	print('______________________________')

	alpha, beta, theta = -alpha * np.pi / 180, -beta * np.pi / 180, -theta * np.pi / 180
	rotation_matrix = [[np.cos(alpha) * np.cos(beta), np.cos(alpha) * np.sin(beta) * np.sin(theta) - np.sin(alpha) * np.cos(theta), np.cos(alpha) * np.sin(beta) * np.cos(theta) + np.sin(alpha) * np.sin(theta)]
	                 ,[np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta) * np.sin(theta) + np.cos(alpha) * np.cos(theta), np.sin(alpha) * np.sin(beta) * np.sin(theta) - np.cos(alpha) * np.sin(theta)]
	                 ,[-np.sin(beta), np.cos(beta) * np.sin(theta), np.cos(beta) * np.cos(theta)]]

	x_rotation = [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]]
	y_rotation = [[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]]
	z_rotation = [[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]]


	return np.matmul(matrix, z_rotation) 

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

def getAngle(side, hypo):
	return np.arccos(side / hypo)

def normalize_angle(df):
	for i in range(len(df) - 1):
		side = df[i + 1].time - df[i].time
		xx, xy, xz = df[i + 1]['x(t)(x)'] - df[i]['x(t)(x)'], df[i + 1]['x(t)(y)'] - df[i]['x(t)(y)'], df[i + 1]['x(t)(z)'] - df[i]['x(t)(z)']
		angleX, angleY, angleZ = getAngle(side, xx), getAngle(side, xy), getAngle(side, xz)