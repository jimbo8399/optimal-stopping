# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

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
from linreg.lin_reg_model import get_linear_regression_model as get_model
from linreg.lin_reg_model import k_fold_cv as get_error
from policies.policy import *

W = int(sys.argv[1]) # window size
policies = {"policyE":policyE, "policyN":policyN, "policyM":policyM, "policyA":policyA, "policyC":policyC, "policyR":policyR}
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
    sensor = all_sensors[sensor_ind].iloc[:200,:]

    dataset_length = len(sensor)

    if dataset_length<W:
        print("insufficient amount of data")
        exit(1)

    def getNewX(data):
        return data.temperature.values.reshape(-1,1)

    def getNewY(data, S=None):
        return data.humidity.values.reshape(-1,1)

    if policyName=="policyM":
        err_diff, err_storage, init_err = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY, alpha=1)
    else:
        err_diff, err_storage, init_err = applyPolicy(W, sensor, get_model, get_error, getNewX, getNewY)

    '''
    Plot Error rate difference
    '''
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xticks(tuple(range(1,len(err_storage)+15,15)))
  
    plt.plot(range(1,len(err_diff)+1), err_diff, fillstyle='bottom')

    plt.xlim(left=0)
    plt.ylim(bottom=0)
  
    plt.xlabel("Window index")
    plt.ylabel("Error rate difference, |e-e'|")
    plt.title("Absolute error difference for SUV sensor ["+sensor_names[sensor_ind]+"], w="+str(W)+",\nusing Linear Regression and "+policyName)

    plt.tight_layout()

    plt.savefig('results/dataset_2_lin_reg/'+policyName+'/abs_err_diff_'+sensor_names[sensor_ind]+'_w_'+str(W)+'.png')
    
    '''
    Plot histogram of |e-e'|
    '''
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(err_diff,color='xkcd:azure',bins=(200-W)//2, edgecolor='black')

    median = np.median(err_diff)

    props = dict(boxstyle='round', facecolor='white')
    for i in range(1,len(patches)):
        if patches[i-1].xy[0]<=median and median<patches[i].xy[0]:
            patches[i-1].set_color('xkcd:banana')
            patches[i-1].set_edgecolor('black')
            patches[i-1].set_hatch('/')
            ax.text(patches[i-1].xy[0]+patches[i-1].get_width()/2,n[i-1],"median:\n{0:1.3f}".format(median),va='center', color='r', bbox=props)
    if patches[i].xy[0]<=median:
        patches[i].set_color('xkcd:banana')
        patches[i].set_edgecolor('black')
        patches[i].set_hatch('/')
        ax.text(patches[i].xy[0]+patches[i].get_width()/2,n[i-1],"median:\n{0:1.3f}".format(median),va='center', color='r', bbox=props)

    plt.xlabel("Error rate difference, |e-e'|")
    plt.ylabel("Frequency")
    plt.title("Absolute error difference for SUV sensor ["+sensor_names[sensor_ind]\
        +"], w="+str(W)+",\nusing Linear Regression and the corresponding median for the data")

    fig.tight_layout()
    
    plt.savefig('results/dataset_2_lin_reg/'+policyName+'/hist_data_dist_median_'+sensor_names[sensor_ind]+'_w_'+str(W)+'.png')

    '''
    Plot all error rates
    '''
    fig, ax = plt.subplots()
  
    plt.plot(range(0,len(err_storage)), err_storage[0:], fillstyle='bottom')
  
    ax.hlines(init_err,0,len(err_storage)-1,colors='r')
    props = dict(boxstyle='round', facecolor='white')
    ax.text(len(err_storage)-0.5,init_err,"{0:f}".format(init_err),va='center', color='r', bbox=props)
    ax.grid(True)
    ax.set_xticks(tuple(range(0,len(err_storage)+1,2)))
  
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel("Window index")
    plt.ylabel("Error rate, e")
    plt.title("Error rate increase/decrease compared\nto initial error rate for SUV sensor ["+sensor_names[sensor_ind]+"], w="+str(W)+",\nusing Linear Regression")
  
    plt.tight_layout()

    plt.savefig('results/dataset_2_lin_reg/'+policyName+'/err_rates_'+sensor_names[sensor_ind]+'_w_'+str(W)+'.png')
