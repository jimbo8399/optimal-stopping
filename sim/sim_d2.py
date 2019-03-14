# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pickle

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

# Add the bin folder to PATH
bin_path = proj_path / "bin"
sys.path.append(proj_pathname)

from bin.data_import import data_init
from bin.data_import import import_dataset_2 as im
from bin.plot_d2 import *
from linreg.lin_reg_model import get_linear_regression_model as get_model
from linreg.lin_reg_model import k_fold_cv as get_error
from policies.policy import *
from bin.result import Result
from bin.b_penalty_processing import calc_t

SIZE = 350

W = int(sys.argv[1]) # window size
policies = {"policyE":policyE, 
            "policyN":policyN, 
            "policyM":policyM, 
            "policyA":policyA, 
            "policyC":policyC, 
            "policyR":policyR, 
            "policyC":policyC,
            "policyOST":policyOST
}
policyName = sys.argv[3]
if policyName=='policyOST' and len(sys.argv)==5:
    ostPenalty = int(sys.argv[4])
else:
    ostPenalty = -1

if policyName=='policyR' and len(sys.argv)==5:
    probR = int(sys.argv[4])
else:
    probR = 10

applyPolicy = policies.get(policyName)
if not callable(applyPolicy):
    print("Nonexistent policy")
    exit(0)

# Initialising data structure
data_init()
# sensor_names = ["pi2","pi3","pi4","pi5"]
sensor_name = sys.argv[2]
sensor_data = im(sensor_name)

# Import data from each Dataset, USV=pi2, pi3, pi4, pi5
# Getting only 60 datapoints
sensor = sensor_data.iloc[:SIZE,:]

dataset_length = len(sensor)

if dataset_length<W:
    print("insufficient amount of data")
    exit(1)

def getNewX(data):
    return data.temperature.values.reshape(-1,1)

def getNewY(data, S=None):
    return data.humidity.values.reshape(-1,1)

print("Sensor "+sensor_name)
if policyName=="policyM":
    err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, alpha=0.5)
elif policyName=="policyC":
    err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, cusumT=2)
elif policyName=="policyOST":
    err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, theta = 3, B = ostPenalty)
elif policyName=="policyR":
    sensor = sensor_data.iloc[101:SIZE,:]
    err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, probR=probR)
else:
    sensor = sensor_data.iloc[101:SIZE,:]
    err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY)

waiting_time = calc_t(comm)

result = Result(sensor_name, 
    err_diff, 
    err_storage, 
    comm, 
    policyName, 
    W, 
    init_err,
    SIZE,
    waiting_time=waiting_time,
    penalty_b=ostPenalty,
    dataset='d2'
    )

if ostPenalty == -1:
    pickle.dump(result, open('results/raw_data/waiting_time_'+policyName+'_d2_'+sensor_name+"_"+str(W)+'.pkl','wb'))
else:
    pickle.dump(result,open('results/raw_data/waiting_time_'+policyName+'_'+'ostpenalty_'+str(ostPenalty)+'_d2_'+sensor_name+"_"+str(W)+'.pkl','wb'))

pickle.dump(result, open("results/raw_data/results_d2_"+policyName+'_'+sensor_name+'_'+str(W)+'.pkl', 'wb'))