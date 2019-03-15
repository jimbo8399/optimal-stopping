import os
from pathlib import Path
import numpy as np
from scipy import stats
from bin.result_processing import loadFile
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

results_path = Path("results/raw_data")
files = os.listdir(results_path)

d1_data = {}
d2_data = {}

for filename in files:
	if filename[:7]=='results':
		with open(results_path/filename,"rb") as f:
			dataset, result = loadFile(f)
			if result.policyName != 'policyN':
				if dataset == 1:
					if result.sensor_name not in d1_data:
						d1_data[result.sensor_name] = []
					data = [(result.policyName, err) for err in result.err_diff]
					d1_data[result.sensor_name] += data
				elif dataset == 2:
					if result.sensor_name not in d2_data:
						d2_data[result.sensor_name] = []
					data = [(result.policyName, err) for err in result.err_diff]
					d2_data[result.sensor_name] += data

print("===For Dataset 1 using SVR with an RBF kernel===")
for sensor_name in d1_data:
	data = np.rec.array(d1_data[sensor_name], dtype = [('Policy','|U10'),('ErrorDiff', '<f8')])
	 
	f, p = stats.f_oneway(data[data['Policy'] == 'policyC'].ErrorDiff,
	                      data[data['Policy'] == 'policyOST'].ErrorDiff,
	                      data[data['Policy'] == 'policyA'].ErrorDiff,
	                      data[data['Policy'] == 'policyR'].ErrorDiff,
	                      data[data['Policy'] == 'policyM'].ErrorDiff,
	                      data[data['Policy'] == 'policyE'].ErrorDiff)
	 
	print ('One-way ANOVA')
	 
	print ('F value:', f)
	if p <= 0.05:
		print ('P value: {} <= 0.05'.format(p))
		print("=> Reject H0\n")
	else:
		print ('P value: {} > 0.05'.format(p))
		print("=> Fail to reject H0\n")

	'''
	Perform Tukey T-Test
	'''

	mc = MultiComparison(data['ErrorDiff'], data['Policy'])
	result = mc.tukeyhsd(alpha=0.05)
	with open('results/svr_rbf_'+sensor_name+'_error_plot_diff_means.txt', 'w') as f:
		f.write(str(result.summary()))
	 
	result.plot_simultaneous(comparison_name='policyE')
	plt.savefig('results/svr_rbf_'+sensor_name+'_error_plot_diff_means'+'.png')
print("================================================")

print("\n\n===For Dataset 2 using Linear Regression===")
for sensor_name in d2_data:
	data = np.rec.array(d2_data[sensor_name], dtype = [('Policy','|U10'),('ErrorDiff', '<f8')])
	f, p = stats.f_oneway(data[data['Policy'] == 'policyC'].ErrorDiff,
	                      data[data['Policy'] == 'policyOST'].ErrorDiff,
	                      data[data['Policy'] == 'policyA'].ErrorDiff,
	                      data[data['Policy'] == 'policyR'].ErrorDiff,
	                      data[data['Policy'] == 'policyM'].ErrorDiff,
	                      data[data['Policy'] == 'policyE'].ErrorDiff)
	 
	print ('One-way ANOVA')
	 
	print ('F value:', f)
	if p <= 0.05:
		print ('P value: {} <= 0.05'.format(p))
		print("=> Reject H0\n")
	else:
		print ('P value: {} > 0.05'.format(p))
		print("=> Fail to reject H0\n")

	'''
	Perform Tukey T-Test
	'''

	mc = MultiComparison(data['ErrorDiff'], data['Policy'])
	result = mc.tukeyhsd(alpha=0.05)
	with open('results/lin_reg_'+sensor_name+'_error_plot_diff_means.txt', 'w') as f:
		f.write(str(result.summary()))
	 
	result.plot_simultaneous(comparison_name='policyE')
	plt.savefig('results/lin_reg_'+sensor_name+'_error_plot_diff_means'+'.png')
print("\n\n===========================================")
