import os
from pathlib import Path
import numpy as np
from scipy import stats
from bin.result_processing import loadFile

results_path = Path("results/raw_data")
files = os.listdir(results_path)

d1_all_waiting_times = []
d1_all_penalties = []
d1_sensor_name = None
d1_window_size = None
d1_data = []

d2_all_waiting_times = []
d2_all_penalties = []
d2_sensor_name = None
d2_window_size = None
d2_data = []


for filename in files:
	if filename[:12]=='waiting_time':
		with open(results_path/filename,"rb") as f:
			dataset, result = loadFile(f)
			if result.policyName != 'policyE' and result.policyName != 'policyN':
				if dataset == 1:
					data = [(result.policyName, wt) for wt in result.waiting_time]
					d1_data += data
				elif dataset == 2:
					data = [(result.policyName, wt) for wt in result.waiting_time]
					d2_data += data

print("===For Dataset 1 using SVR with an RBF kernel===")
data = np.rec.array(d1_data, dtype = [('Policy','|U10'),('Waiting', '<i8')])
 
f, p = stats.f_oneway(data[data['Policy'] == 'policyA'].Waiting,
                      data[data['Policy'] == 'policyC'].Waiting,
                      data[data['Policy'] == 'policyM'].Waiting,
                      data[data['Policy'] == 'policyR'].Waiting,
                      data[data['Policy'] == 'policyOST'].Waiting)
 
print ('One-way ANOVA')
print ('=============')
 
print ('F value:', f)
print ('P value: {}'.format(p))
if p <= 0.5:
	print("=> Reject H0\n")
else:
	print("=> Fail to reject H0\n")

'''
Perform Tukey T-Test
'''

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

mc = MultiComparison(data['Waiting'], data['Policy'])
result = mc.tukeyhsd()
 
print(result)
print(mc.groupsunique)

print("\n\n===For Dataset 2 using Linear Regression===")
data = np.rec.array(d2_data, dtype = [('Policy','|U10'),('Waiting', '<i8')])
f, p = stats.f_oneway(data[data['Policy'] == 'policyA'].Waiting,
                      data[data['Policy'] == 'policyC'].Waiting,
                      data[data['Policy'] == 'policyM'].Waiting,
                      data[data['Policy'] == 'policyR'].Waiting,
                      data[data['Policy'] == 'policyOST'].Waiting)
 
print ('One-way ANOVA')
print ('=============')
 
print ('F value:', f)
print ('P value: {}'.format(p))
if p <= 0.5:
	print("=> Reject H0\n")
else:
	print("=> Fail to reject H0\n")

'''
Perform Tukey T-Test
'''

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

mc = MultiComparison(data['Waiting'], data['Policy'])
result = mc.tukeyhsd()
 
print(result)
print(mc.groupsunique)
