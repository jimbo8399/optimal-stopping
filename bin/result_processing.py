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

def plotBoxPlotWaitingTimeDataset1(wt, B, w, S):
	import bin.plot_d1 as plt1
	plt1.plotBoxPlotsForWaitingTime(wt, B, w, S)

def plotBoxPlotWaitingTimeDataset2(wt, B, w, S):
	import bin.plot_d2 as plt2
	plt2.plotBoxPlotsForWaitingTime(wt, B, w, S)

def plotAllResults():
	import os
	from pathlib import Path
	results_path = Path("results/raw_data")
	files = os.listdir(results_path)

	d1_all_waiting_times = []
	d1_all_penalties = []
	d1_sensor_name = None
	d1_window_size = None
	d2_all_waiting_times = []
	d2_all_penalties = []
	d2_sensor_name = None
	d2_window_size = None


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
					if result.policyName == "policyOST":
						d1_all_waiting_times.append(result.waiting_time)
						d1_all_penalties.append(result.penalty_b)
						d1_sensor_name = result.sensor_name
						d1_window_size = result.w
				elif dataset == 2:
					plotWaitingTimeDataset2(result)
					if result.policyName == "policyOST":
						d2_all_waiting_times.append(result.waiting_time)
						d2_all_penalties.append(result.penalty_b)
						d2_sensor_name = result.sensor_name
						d2_window_size = result.w

	# plotBoxPlotWaitingTimeDataset1(d1_all_waiting_times,
	# 							d1_all_penalties,
	# 							d1_window_size,
	# 							d1_sensor_name
	# 							)
	# plotBoxPlotWaitingTimeDataset2(d2_all_waiting_times,
	# 							d2_all_penalties,
	# 							d2_window_size,
	# 							d2_sensor_name
	# 							)