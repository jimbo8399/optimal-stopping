import pickle

def loadFile(filename):
	result = pickle.load(filename)
	if result.dataset == 'd2':
		return 2, result
	elif result.dataset == 'd1':
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
	import bin.plot_d1 as plt1
	plt1.plotHistForWaitingTime(res.waiting_time, 
		res.penalty_b, 
		res.kernel_name, 
		res.policyName, 
		res.kernel_dir, 
		res.w, 
		res.sensor_name, 
		res.size
	)

def plotWaitingTimeDataset2(res):
	import bin.plot_d2 as plt2
	plt2.plotHistForWaitingTime(res.waiting_time,
		res.penalty_b, 
		res.policyName, 
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
					plotForDataset2(result)
		elif filename[:12]=='waiting_time':
			with open(results_path/filename,"rb") as f:
				dataset, result = loadFile(f)
				if dataset == 1:
					plotWaitingTimeDataset1(result)
				elif dataset == 2:
					plotWaitingTimeDataset2(result)

