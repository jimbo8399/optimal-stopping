# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pickle
import pandas as pd

PROJ_NAME = "optimal-stopping"

# Locate the Project directory
curr_dir = str(Path.cwd())
start = curr_dir.find(PROJ_NAME)
if start < 0:
	print("ERROR: Project directory not found")
	print("Make sure you have the correct project structure")
	print("and run the simulation from within the project")
proj_pathname = curr_dir[:(start+len(PROJ_NAME))]

# Create path to the project directory
proj_path = Path(proj_pathname)

# Add the project folder to PATH
sys.path.append(proj_pathname)

from bin.data_import import data_init
from bin.data_import import import_dataset_1 as im
from bin.plot_d1 import *
from svr.svr_model import k_fold_cv as get_error
from policies.policy import *
from bin.result import Result

# 100 datapoints are used for the median delay policy, 
# and all policies start using the data from the 100th datapoint
SIZE = 300
W = int(sys.argv[1]) # window size
S = sys.argv[2] # sensor name, choices: R1- R8
if sys.argv[3]=='lin':
	kernel_name = 'Linear'
	kernel_dir = 'lin'
	from svr.svr_model import get_svr_lin_model as get_model
elif sys.argv[3]=='rbf':
	kernel_name = 'RBF'
	kernel_dir = 'rbf'
	from svr.svr_model import get_svr_rbf_model as get_model

policies = {"policyE":policyE, 
			"policyN":policyN, 
			"policyM":policyM, 
			"policyA":policyA, 
			"policyC":policyC, 
			"policyR":policyR, 
			"policyC":policyC
}
policyName = sys.argv[4]
applyPolicy = policies.get(policyName)
if not callable(applyPolicy):
	print("Nonexistent policy")
	exit(0)

# Initialising data structure
data_init()
sensor_dataset = im().iloc[0:SIZE,:]

# plt.plot(sensor_dataset.loc[:,"R1"])

dataset_length = len(sensor_dataset)

if dataset_length<100+W:
	print("insufficient amount of data")
	exit(1)

'''
TODO: mae a method for introducing change in the distribution
'''
### FIRST Artificial change
change = []
### the start and end index is included in df slicing
startind = len(sensor_dataset)//4 + 1
endind = len(sensor_dataset)
changeSize = endind-startind
###
offs = 0.005
for col in sensor_dataset.columns.values:
	if col == 'id' or col == "time":
		emptyEntry = np.zeros((changeSize,1))
		change += [pd.DataFrame(emptyEntry, columns=[col])]
	if col != 'id' and col != "time":
		mean = np.mean(sensor_dataset[col])
		changeUp = np.linspace(0,mean*offs, changeSize/2, endpoint=False)
		changeDown = np.linspace(mean*offs,0, changeSize/2, endpoint=False)
		singleChange = np.concatenate([changeUp,changeDown]).reshape(changeSize,1)
		change += [pd.DataFrame(singleChange, columns=[col])]

change = pd.concat(change, axis=1).values
sample = sensor_dataset.loc[startind:endind].values
amendedSample = sample+change

sensor_dataset = pd.concat([sensor_dataset.loc[:startind-1],
	pd.DataFrame(amendedSample, columns=sensor_dataset.columns.values),
	sensor_dataset.loc[endind+1:]],
	ignore_index=True)

### SECOND Artificial change
change = []
### the start and end index is included in df slicing
startind = len(sensor_dataset)//2
endind = len(sensor_dataset)
changeSize = endind-startind
###
offs = 0.005
for col in sensor_dataset.columns.values:
	if col == 'id' or col == "time":
		emptyEntry = np.zeros((changeSize,1))
		change += [pd.DataFrame(emptyEntry, columns=[col])]
	if col != 'id' and col != "time":
		mean = np.mean(sensor_dataset[col])
		changeUp = np.linspace(0,mean*offs, changeSize/2, endpoint=False)
		changeDown = np.linspace(mean*offs,0, changeSize/2, endpoint=False)
		singleChange = np.concatenate([changeUp,changeDown]).reshape(changeSize,1)
		change += [pd.DataFrame(singleChange, columns=[col])]

change = pd.concat(change, axis=1).values
sample = sensor_dataset.loc[startind:endind].values
amendedSample = sample+change

sensor_dataset = pd.concat([sensor_dataset.loc[:startind-1],
	pd.DataFrame(amendedSample, columns=sensor_dataset.columns.values),
	sensor_dataset.loc[endind+1:]],
	ignore_index=True)

# plt.plot(sensor_dataset.loc[:,"R1"])

# plt.show()

def getNewX(data):
	return data[['Temp.','Humidity']].values

def getNewY(data, S):
	return data[[S]].values

if policyName=="policyM":
	err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S, alpha=0.5)
elif policyName=="policyC":
	err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S, cusumT=0.5)
else:
	sensor_dataset = im().iloc[100:SIZE,:]
	err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S)

result = Result(S,
	err_diff,
	err_storage,
	comm,
	policyName,
	W,
	init_err,
	SIZE,
	kernel_name,
	kernel_dir
	)

pickle.dump(result, open("results/raw_data/results_d1_"+kernel_dir+"_"+S+"_"+policyName+"_"+str(W)+".pkl","wb"))
