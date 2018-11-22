import numpy as np

'''
Policy N: never send the model
'''
def policyN(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S = ""):
	data = sensor_dataset.iloc[0:W,:]

	# Reshape the temperature and humidity values
	init_X = getNewX(data)
	# Reshape the sensor values
	init_y = getNewY(data, S)
	# Build a model to be sent to the Edge Gate
	model = get_model(init_X, init_y)
	# Evaluate the model
	err = get_error(model, init_X, init_y)

	err_diff = []
	err_storage = [err]
	init_err = err
	init_model = model

	comm_count = 1
	comm = [comm_count]

	dataset_length = len(sensor_dataset)
	i = 1
	while (i + W) <= dataset_length:
		# Receive a new datapoint
		data = sensor_dataset.iloc[i:i+W,:]
		X = getNewX(data)
		y = getNewY(data, S)
		# Build a new model with the newly arrived datapoint 
		# and the discarded oldest datapoint
		new_model = get_model(X, y)
		# Evaluate
		new_err = get_error(new_model, X, y)
		err_storage += [new_err]

		init_model_err = get_error(init_model, X, y)
		err_diff += [abs(init_model_err-new_err)]

		# Slide the window with 1
		i += 1

	return err_diff, err_storage, init_err, comm

'''
Policy E: always send the model
'''
def policyE(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S = ""):
	data = sensor_dataset.iloc[0:W,:]

	# Reshape the temperature and humidity values
	init_X = getNewX(data)
	# Reshape the sensor values
	init_y = getNewY(data, S)
	# Build a model to be sent to the Edge Gate
	model = get_model(init_X, init_y)
	# Evaluate the model
	err = get_error(model, init_X, init_y)

	err_diff = []
	err_storage = [err]
	init_err = err
	init_model = model

	comm_count = 1
	comm = [comm_count]

	dataset_length = len(sensor_dataset)
	i = 1
	while (i + W) <= dataset_length:
		# Receive a new datapoint
		data = sensor_dataset.iloc[i:i+W,:]
		X = getNewX(data)
		y = getNewY(data, S)
		# Build a new model with the newly arrived datapoint 
		# and the discarded oldest datapoint
		new_model = get_model(X, y)
		# Evaluate
		new_err = get_error(new_model, X, y)
		err_storage += [new_err]

		init_model_err = get_error(init_model, X, y)
		err_diff += [abs(init_model_err-new_err)]
		
		init_model = new_model
		comm_count+=1
		comm += [comm_count]

		# Slide the window with 1
		i += 1

	return err_diff, err_storage, init_err, comm

'''
Policy M: send model only when error diff is above the median error of first 100 error diffs
'''
def policyM(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S = "", alpha=0.5):
	if len(sensor_dataset) < 100:
		print("Insufficient ammount of data to compute the error rate median, has to be at least 100 datapoints")
		exit(0)
	else:
		err_diff, _, _, _ = policyN(W, sensor_dataset.iloc[0:100,:], get_model, get_error, getNewX, getNewY, S)
	median = np.median(err_diff)

	sensor_dataset = sensor_dataset.iloc[100:,:]

	data = sensor_dataset.iloc[0:W,:]

	# Reshape the temperature and humidity values
	init_X = getNewX(data)
	# Reshape the sensor values
	init_y = getNewY(data, S)
	# Build a model to be sent to the Edge Gate
	model = get_model(init_X, init_y)
	# Evaluate the model
	err = get_error(model, init_X, init_y)

	err_diff = []
	err_storage = [err]
	init_err = err
	init_model = model

	comm_count = 1
	comm = [comm_count]

	dataset_length = len(sensor_dataset)
	i = 1
	while (i + W) <= dataset_length:

		#Update median every 100 datapoints
		if i%100==0:
		# 	print("Window:", W, "for sensor:",S)
		# 	print("Median before: %1.10f"%median)
			median = np.median(err_diff[-100:])
		# 	print("Median after: %1.10f \n"%median)

		# Receive a new datapoint
		data = sensor_dataset.iloc[i:i+W,:]
		X = getNewX(data)
		y = getNewY(data, S)
		# Build a new model with the newly arrived datapoint 
		# and the discarded oldest datapoint
		new_model = get_model(X, y)
		# Evaluate
		new_err = get_error(new_model, X, y)
		err_storage += [new_err]

		init_model_err = get_error(init_model, X, y)
		diff = abs(init_model_err-new_err)
		err_diff += [diff]
		if diff > median*alpha:
			init_model = new_model
			comm_count += 1
		comm += [comm_count]

		# Slide the window with 1
		i += 1

	return err_diff, err_storage, init_err, comm

'''
Policy A: send model only when the new model is more accurate than the one at the edge gate
'''
def policyA(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S = ""):
	data = sensor_dataset.iloc[0:W,:]

	# Reshape the temperature and humidity values
	init_X = getNewX(data)
	# Reshape the sensor values
	init_y = getNewY(data, S)
	# Build a model to be sent to the Edge Gate
	model = get_model(init_X, init_y)
	# Evaluate the model
	err = get_error(model, init_X, init_y)

	err_diff = []
	err_storage = [err]
	init_err = err
	init_model = model

	comm_count = 1
	comm = [comm_count]

	dataset_length = len(sensor_dataset)
	i = 1
	while (i + W) <= dataset_length:
		# Receive a new datapoint
		data = sensor_dataset.iloc[i:i+W,:]
		X = getNewX(data)
		y = getNewY(data, S)
		# Build a new model with the newly arrived datapoint 
		# and the discarded oldest datapoint
		new_model = get_model(X, y)
		# Evaluate
		new_err = get_error(new_model, X, y)
		err_storage += [new_err]

		init_model_err = get_error(init_model, X, y)
		diff = abs(init_model_err-new_err)
		err_diff += [diff]
		if init_model_err > new_err:
			init_model = new_model
			comm_count += 1
		comm += [comm_count]

		# Slide the window with 1
		i += 1

	return err_diff, err_storage, init_err, comm

def policyC():
	return 0

'''
Policy R: send model after a random number of received datapoints
'''
def policyR(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S = ""):
	data = sensor_dataset.iloc[0:W,:]

	# Reshape the temperature and humidity values
	init_X = getNewX(data)
	# Reshape the sensor values
	init_y = getNewY(data, S)
	# Build a model to be sent to the Edge Gate
	model = get_model(init_X, init_y)
	# Evaluate the model
	err = get_error(model, init_X, init_y)

	err_diff = []
	err_storage = [err]
	init_err = err
	init_model = model

	comm_count = 1
	comm = [comm_count]

	dataset_length = len(sensor_dataset)
	# Generate a list of random length with random unique waiting times
	random_waiting = set(np.random.randint(1+W, dataset_length, 10))
	i = 1
	while (i + W) <= dataset_length:
		# Receive a new datapoint
		data = sensor_dataset.iloc[i:i+W,:]
		X = getNewX(data)
		y = getNewY(data, S)
		# Build a new model with the newly arrived datapoint 
		# and the discarded oldest datapoint
		new_model = get_model(X, y)
		# Evaluate
		new_err = get_error(new_model, X, y)
		err_storage += [new_err]

		init_model_err = get_error(init_model, X, y)
		err_diff += [abs(init_model_err-new_err)]
		if i in random_waiting:
			init_model = new_model
			comm_count += 1
		comm += [comm_count]

		# Slide the window with 1
		i += 1

	return err_diff, err_storage, init_err, comm