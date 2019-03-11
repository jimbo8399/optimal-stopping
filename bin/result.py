def calc_waiting_time(comm):
	    waiting = {}
	    i = 0
	    while i < len(comm):
	        waiting[comm[i]] = waiting.get(comm[i],0)+1
	        i += 1
	    return [waiting[k] for k in sorted(waiting.keys())]

class Result:
	
	def __init__(self, 
		sensor_name, 
		err_diff, 
		err_storage, 
		communication, 
		policyName, 
		window_size, 
		init_error,
		size,
		kernel_name="",
		kernel_dir="",
		waiting_time=[],
		penalty_b=-1,
		dataset=""
		):
		self.sensor_name = sensor_name
		self.err_diff = err_diff
		self.err_storage = err_storage
		self.comm = communication
		self.policyName = policyName
		self.w = window_size
		self.init_error = init_error
		self.size = size
		self.waiting_time = calc_waiting_time(communication)
		self.kernel_name = kernel_name
		self.kernel_dir = kernel_dir
		self.waiting_time = waiting_time
		self.penalty_b = penalty_b
		self.dataset = dataset

		