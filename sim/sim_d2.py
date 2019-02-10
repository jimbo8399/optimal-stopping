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

SIZE = 275

W = int(sys.argv[1]) # window size
policies = {"policyE":policyE, 
            "policyN":policyN, 
            "policyM":policyM, 
            "policyA":policyA, 
            "policyC":policyC, 
            "policyR":policyR, 
            "policyC":policyC,
            "policyOST":policyOST,
            "policyCostAware":policyCostAware
}
policyName = sys.argv[2]
applyPolicy = policies.get(policyName)
if not callable(applyPolicy):
    print("Nonexistent policy")
    exit(0)

# Initialising data structure
data_init()
# sensor_names = ["pi2","pi3","pi4","pi5"]
sensor_names = ["pi3"]
all_sensors = im(sensor_names)

# Results to be dumped into a pickle file
results = []

# Import data from each Dataset, USV=pi2, pi3, pi4, pi5
# Getting only 60 datapoints
for sensor_ind in range(len(all_sensors)):
    sensor = all_sensors[sensor_ind].iloc[:SIZE,:]

    dataset_length = len(sensor)

    if dataset_length<W:
        print("insufficient amount of data")
        exit(1)

    def getNewX(data):
        return data.temperature.values.reshape(-1,1)

    def getNewY(data, S=None):
        return data.humidity.values.reshape(-1,1)

    print("Sensor "+sensor_names[sensor_ind])
    if policyName=="policyM":
        err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, alpha=0.5)
    elif policyName=="policyC":
        err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, cusumT=5)
    elif policyName=="policyOST":
        err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, theta = 3, B = 5)
    elif policyName=="policyCostAware":
        sensor = all_sensors[sensor_ind].iloc[101:SIZE,:]
        err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, alpha = 0.005)
    else:
        sensor = all_sensors[sensor_ind].iloc[101:SIZE,:]
        err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY)

    result = Result(sensor_names[sensor_ind], 
        err_diff, 
        err_storage, 
        comm, 
        policyName, 
        W, 
        init_err,
        SIZE
        )
    results.append(result)

pickle.dump(results, open("results/raw_data/results_d2_"+policyName+"_"+str(W)+".pkl", "wb"))