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
		kernel_dir=""
		):
		self.sensor_name = sensor_name
		self.err_diff = err_diff
		self.err_storage = err_storage
		self.comm = communication
		self.policyName = policyName
		self.w = window_size
		self.init_error = init_error
		self.size = size
		self.kernel_name = kernel_name
		self.kernel_dir = kernel_dir