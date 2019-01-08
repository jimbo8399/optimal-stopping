# -*- coding: utf-8 -*-
import sys
from pathlib import Path

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

SIZE = 275
W = int(sys.argv[1]) # window size
policies = {"policyE":policyE, "policyN":policyN, "policyM":policyM, "policyA":policyA, "policyC":policyC, "policyR":policyR, "policyC":policyC}
policyName = sys.argv[2]
applyPolicy = policies.get(policyName)
if not callable(applyPolicy):
    print("Nonexistent policy")
    exit(0)

# Initialising data structure
data_init()
all_sensors = im()
sensor_names = ["pi2","pi3","pi4","pi5"]
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

    if policyName=="policyM":
        err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, alpha=0.5)
    else:
        sensor = all_sensors[sensor_ind].iloc[100:SIZE,:]
        err_diff, err_storage, init_err, comm = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY)

    plotErrorRateDiff(err_diff, comm, policyName, W, sensor_names[sensor_ind])
    plotHistErr(err_diff, policyName, W, sensor_names[sensor_ind], SIZE)
    plotErrRate(err_storage, init_err, policyName, W, sensor_names[sensor_ind])