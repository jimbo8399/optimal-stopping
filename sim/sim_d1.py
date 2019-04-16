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
from bin.b_penalty_processing import calc_t

# 100 datapoints are used for the median delay policy, 
# and all policies start using the data from the 101st datapoint
SIZE = 350

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
			"policyC":policyC,
			"policyOST":policyOST
}
policyName = sys.argv[4]
if policyName=='policyOST' and len(sys.argv)==6:
	ostPenalty = int(sys.argv[5])
else:
	ostPenalty = -1

if policyName=='policyR' and len(sys.argv)==6:
	probR = int(sys.argv[5])
else:
	probR = 10

applyPolicy = policies.get(policyName)
if not callable(applyPolicy):
	print("Nonexistent policy")
	exit(0)

# Initialising data structure
data_init()
sensor_dataset = im().iloc[0:SIZE,:]

dataset_length = len(sensor_dataset)

if dataset_length<100+W:
	print("insufficient amount of data")
	exit(1)

### FIRST Artificial change
change = []
### the start and end index is included in df slicing
changeSize = 12
startind = int(len(sensor_dataset)/2)
endind = startind + changeSize
totalLength = len(sensor_dataset) - startind
print("First artificial change from",startind-(100+W))
###
offs = 0.1
for col in sensor_dataset.columns.values:
	if col == 'id' or col == "time"  or col == "Temp." or col == "Humidity":
		emptyEntry = np.zeros((totalLength,1))
		change += [pd.DataFrame(emptyEntry, columns=[col])]
	if col != 'id' and col != "time"  and col != "Temp." and col != "Humidity":
		mean = np.mean(sensor_dataset[col])
		changeUp = np.linspace(0,mean*offs, changeSize, endpoint=False)
		# introduce randomness
		changeUp = [x-np.random.randint(0,10)/1000 for x in changeUp]
		changeStill = np.repeat(changeUp[-1], totalLength-changeSize)
		# introduce randomness
		changeStill = [x+np.random.randint(0,5)/1000 for x in changeStill]
		singleChange = np.concatenate([changeUp,changeStill]).reshape(totalLength,1)
		change += [pd.DataFrame(singleChange, columns=[col])]

change = pd.concat(change, axis=1).values
sample = sensor_dataset.loc[startind:].values
amendedSample = sample+change

sensor_dataset = pd.concat([sensor_dataset.loc[:startind-1],
	pd.DataFrame(amendedSample, columns=sensor_dataset.columns.values)],
	# sensor_dataset.loc[totalLength+1:]],
	ignore_index=True)

### SECOND Artificial change - Outlier-like
change = []
### the start and end index is included in df slicing
changeSize = 4
startind = int(len(sensor_dataset)/1.5)
endind = startind + changeSize
print("Second artificial change form",startind-(100+W),"to",endind-(100+W))
###
offs = 0.025
for col in sensor_dataset.columns.values:
	if col == 'id' or col == "time" or col == "Temp." or col == "Humidity":
		emptyEntry = np.zeros((changeSize,1))
		change += [pd.DataFrame(emptyEntry, columns=[col])]
	if col != 'id' and col != "time" and col != "Temp." and col != "Humidity":
		mean = np.mean(sensor_dataset[col])
		changeUp = np.linspace(0,mean*offs, int(changeSize/2), endpoint=False)
		# introduce randomness
		changeUp = [x-np.random.randint(0,10)/1000 for x in changeUp]
		changeDown = np.linspace(mean*offs,0, int(changeSize/2), endpoint=False)
		# introduce randomness
		changeDown = [x+np.random.randint(0,5)/1000 for x in changeDown]
		singleChange = np.concatenate([changeUp,changeDown]).reshape(changeSize,1)
		change += [pd.DataFrame(singleChange, columns=[col])]

change = pd.concat(change, axis=1).values
sample = sensor_dataset.loc[startind:endind-1].values
amendedSample = sample+change

sensor_dataset = pd.concat([sensor_dataset.loc[:startind-1],
	pd.DataFrame(amendedSample, columns=sensor_dataset.columns.values),
	sensor_dataset.loc[endind+1:]],
	ignore_index=True)

def getNewX(data):
	return data[['Temp.','Humidity']].values

def getNewY(data, S):
	return data[[S]].values

if policyName=="policyM":
	err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S, alpha=0.5)
elif policyName=="policyC":
	err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S, cusumT=0.5)
elif policyName=="policyOST":
    err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S, theta = 1, B = ostPenalty)
elif policyName=="policyR":
	sensor_dataset = im().iloc[101:SIZE,:]
	err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S, probR=probR)
else:
	sensor_dataset = im().iloc[101:SIZE,:]
	err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S)

waiting_time = calc_t(comm)

result = Result(S,
	err_diff,
	err_storage,
	comm,
	policyName,
	W,
	init_err,
	SIZE,
	kernel_name=kernel_name,
	kernel_dir=kernel_dir,
	waiting_time=waiting_time,
	penalty_b=ostPenalty,
	dataset='d1'
	)

if ostPenalty == -1:
    pickle.dump(result, open('results/raw_data/waiting_time_'+policyName+'_d1_'+kernel_dir+'_'+S+'_'+str(W)+'.pkl','wb'))
else:
    pickle.dump(result,open('results/raw_data/waiting_time_'+policyName+'_'+'ostpenalty_'+str(ostPenalty)+'_d1_'+kernel_dir+'_'+S+'_'+str(W)+'.pkl','wb'))

pickle.dump(result, open("results/raw_data/results_d1_"+kernel_dir+"_"+S+"_"+policyName+"_"+str(W)+".pkl","wb"))
