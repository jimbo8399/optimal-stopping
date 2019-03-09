import pickle

def loadFile(filename):
	result = pickle.load(filename)
	if type(result)==type([]):
		return 2, result
	else:
		return 1, result

def plotForDataset1(res):
	import bin.plot_d1 as plt
	plt.plotErrorRateDiff(res.err_diff, 
		res.comm, 
		res.kernel_name, 
		res.policyName, 
		res.kernel_dir, 
		res.w, 
		res.sensor_name
	)
	plt.plotHistErr(res.err_diff, 
		res.kernel_name, 
		res.policyName, 
		res.kernel_dir, 
		res.w, 
		res.sensor_name, 
		res.size
	)
	plt.plotErrRate(res.err_storage, 
		res.init_error, 
		res.kernel_name, 
		res.policyName, 
		res.kernel_dir, 
		res.w, 
		res.sensor_name
	)

def plotForDataset2(res):
	import bin.plot_d2 as plt
	plt.plotErrorRateDiff(res.err_diff, 
		res.comm,  
		res.policyName,  
		res.w, 
		res.sensor_name
	)
	plt.plotHistErr(res.err_diff, 
		res.policyName,  
		res.w, 
		res.sensor_name, 
		res.size
	)
	plt.plotErrRate(res.err_storage, 
		res.init_error,  
		res.policyName,  
		res.w, 
		res.sensor_name
	)

def plotWaitingTimeDataset1(res):
	import bin.plot_d1 as plt
	plt.plotHistForPenaltyB(res.waiting_time, 
		res.penalty_b, 
		res.kernel_name, 
		res.policyName, 
		res.kernel_dir, 
		res.w, 
		res.sensor_name, 
		res.size
	)

def plotAllResults():
	import os
	from pathlib import Path
	results_path = Path("results/raw_data")
	files = os.listdir(results_path)
	for filename in files:
		if filename[:7]=="results":
			with open(results_path/filename,"rb") as f:
				dataset, result = loadFile(f)
				if dataset == 1:
					plotForDataset1(result)
				elif dataset==2:
					for sensor_result in result:
						plotForDataset2(sensor_result)
		elif filename[:10]=='ostpenalty':
			with open(results_path/filename,"rb") as f:
				(dataset, alg, ostPenalty, t) = pickle.load(f)
				print(dataset, alg, ostPenalty, t)

