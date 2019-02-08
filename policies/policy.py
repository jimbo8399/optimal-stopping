import numpy as np
from scipy.stats import gamma

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
	
	BASEN = 100

	if len(sensor_dataset) < BASEN:
		print("Insufficient ammount of data to compute the error rate median, has to be at least 100 datapoints")
		exit(0)
	else:
		err_diff, _, _, _ = policyN(W, sensor_dataset.iloc[0:BASEN,:], get_model, get_error, getNewX, getNewY, S)
	median = np.median(err_diff)

	sensor_dataset = sensor_dataset.iloc[BASEN+1:,:]

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

def policyC(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S = "", cusumT = 3.3):

	BASEN = 75

	if len(sensor_dataset) < BASEN:
		print("Insufficient ammount of data to compute the error rate median, has to be at least 100 datapoints")
		exit(0)
	else:
		err_diff_good, _, _, _ = policyE(W, sensor_dataset.iloc[0:BASEN,:], get_model, get_error, getNewX, getNewY, S)
		err_diff_bad, _, _, _ = policyN(W, sensor_dataset.iloc[0:BASEN,:], get_model, get_error, getNewX, getNewY, S)

	# print("\nGOOOOOOD dist", err_diff_good)
	# print("\nBAAAAAAD dist", err_diff_bad)

	#good distribution
	mean_good = np.mean(err_diff_good)
	var_good = np.var(err_diff_good)
	A1 = (mean_good**2)/var_good
	B1 = var_good/mean_good
	# print("\ngood shape ", A1)
	# print("\ngood scale", B1)

	#bad distribution
	mean_bad = np.mean(err_diff_bad)
	var_bad = np.var(err_diff_bad)
	A2 = (mean_bad**2)/var_bad
	B2 = var_bad/mean_bad
	# print("\nbad shape", A2)
	# print("\nbad scale", B2)

	sensor_dataset = sensor_dataset.iloc[BASEN+1:,:]

	data = sensor_dataset.iloc[0:W,:]
	data_at_1 = sensor_dataset.iloc[1:1+W,:]

	# Reshape the temperature and humidity values
	init_X_at_1 = getNewX(data_at_1)
	init_X = getNewX(data)
	# Reshape the sensor values
	init_y_at_1 = getNewY(data_at_1, S)
	init_y = getNewY(data, S)
	# Build a model to be sent to the Edge Gate
	model_at_1 = get_model(init_X_at_1, init_y_at_1)
	model = get_model(init_X, init_y)
	# Evaluate the model
	err_at_1 = get_error(model_at_1, init_X, init_y)
	err = get_error(model, init_X, init_y)

	diff = abs(err_at_1-err)

	good_diff = gamma.pdf(diff, a=A1, scale=B1)
	bad_diff = gamma.pdf(diff, a=A2, scale=B2)

	err_diff = [diff]
	err_storage = [err, err_at_1]
	init_err = err
	init_model = model

	#log ratio likelihood value going from good to bad distribution
	log_ratio = np.log(bad_diff/good_diff)
	log_sum = log_ratio
	log_ratio_vector = [log_ratio]
	min_sum = min(log_ratio_vector)
	g = [] #decision vector
	g_value = 0 #decision value monitored if it's exceeding the threshold

	comm_count = 1
	comm = [comm_count]

	dataset_length = len(sensor_dataset)
	i = 2
	while (i + W) <= dataset_length:

		# Receive a new datapoint
		data = sensor_dataset.iloc[i:i+W,:]
		X = getNewX(data)
		y = getNewY(data,S)
		# Build a new model with the newly arrived datapoint 
		# and the discarded oldest datapoint
		new_model = get_model(X, y)
		# Evaluate
		new_err = get_error(new_model, X, y)
		err_storage += [new_err]

		init_model_err = get_error(init_model, X, y)
		diff = abs(init_model_err-new_err)
		err_diff += [diff]

		good_diff = gamma.pdf(diff, a=A1, scale=B1)
		bad_diff = gamma.pdf(diff, a=A2, scale=B2)
		
		log_ratio = np.log(bad_diff/good_diff)
		log_sum += log_ratio

		min_sum = min(log_ratio_vector)

		g_value = log_sum-min_sum

		log_ratio_vector += [log_sum]
		g += [g_value]

		if g_value > cusumT:
			init_model = new_model
			comm_count += 1

			#reset the algorithm
			log_sum = log_ratio
			log_ratio_vector = [log_ratio]
			min_sum = min(log_ratio_vector)
			g_value = 0

		comm += [comm_count]

		# Slide the window with 1
		i += 1
	# print("\ntreshold", cusumT)
	# print("\nG values",g)
	# print("\nerror diff:", err_diff)

	return err_diff, err_storage, init_err, comm

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

def getRewardDistribution(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S = "", theta = 1, B = 1):
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

	err_sum = err

	dataset_length = len(sensor_dataset)

	i = 1
	Vreward = 0
	Vreward_vector = []
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

	    err_sum += err_diff[-1]

	    #DECIDE if we should update or not
	    if err_sum <= theta:
	        Vreward += 1
	        Vreward_vector += [Vreward]
	    else:
	        Vreward = -B
	        Vreward_vector += [Vreward]
	        Vreward = 0
	        init_model = new_model
	        err_sum = new_err
	        
	    # Slide the window with 1
	    i += 1
	mean = np.mean(err_diff)
	var = np.var(err_diff)

	shape = mean**2/var
	scale = var/mean

	return gamma(shape,scale=scale)


def policyOST(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S = "", theta = 1, B = 1):
	BASEN = 100

	if len(sensor_dataset) < BASEN:
		print("Insufficient ammount of data to compute the error rate median, has to be at least 100 datapoints")
		exit(0)
	else:
		gamma_dist = getRewardDistribution(W, sensor_dataset.iloc[0:BASEN,:], get_model, get_error, getNewX, getNewY, S, theta, B)

	data = sensor_dataset.iloc[BASEN+1:,:]

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

	err_sum = err

	dataset_length = len(sensor_dataset)

	t = 1
	currentV = []
	futureV = []

	comm_count = 1
	comm = [comm_count]
	
	i = 1
	while (i + W) <= dataset_length:

		currentV += [t]
		futureV += [gamma_dist.cdf(theta-err_sum)*(t+1+B)-B]

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

		#DECIDE if we should update or not
		if gamma_dist.cdf(theta-err_sum)<(t+B)/(t+1+B):
			init_model = new_model
			err_sum = err
			comm_count += 1
			t = 0
		else:
			err_sum += err_diff[-1]

		comm += [comm_count]
		t += 1

		# Slide window with 1
		i += 1

	return err_diff, err_storage, init_err, comm