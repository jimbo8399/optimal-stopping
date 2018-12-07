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

# Add the project folder to PATH
sys.path.append(proj_pathname)

from bin.data_import import data_init
from bin.data_import import import_dataset_1 as im
from svr.svr_model import k_fold_cv as get_error
from policies.policy import *

# 100 datapoints are used for the median delay policy, 
# and all policies start using the data from the 100th datapoint
SIZE = 275
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

policies = {"policyE":policyE, "policyN":policyN, "policyM":policyM, "policyA":policyA, "policyC":policyC, "policyR":policyR, "policyC":policyC}
policyName = sys.argv[4]
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

def getNewX(data):
	return data[['Temp.','Humidity']].values

def getNewY(data, S):
	return data[[S]].values

if policyName=="policyM":
	err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S, alpha=0.5)
else:
	sensor_dataset = im().iloc[100:SIZE,:]
	err_diff, err_storage, init_err, comm = applyPolicy(W, sensor_dataset, get_model, get_error, getNewX, getNewY, S)

'''
Plot Error rate difference
'''
fig, ax1 = plt.subplots()
ax1.grid(True)
# ax1.set_xticks(tuple([1]+list(range(15,len(err_diff)+15,15))))
ax1.tick_params(axis="y", labelcolor="C0")
ax1.plot(range(1,len(err_diff)+1), err_diff, fillstyle='bottom')

# instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx() 
ax2.tick_params(axis="y", labelcolor="xkcd:red orange")
ax2.plot(range(0,len(comm)), comm, fillstyle='bottom', color="xkcd:red orange")

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.xlabel("Window index")
ax1.set_ylabel("Error rate difference, |e-e'|", color="C0")
ax2.set_ylabel('Communication rate', color="xkcd:red orange")
plt.title("Absolute error difference for HT sensor system,"+ \
	" s="+ S +", w="+str(W)+\
	",\nusing Support Vector Regression with "+kernel_name+" Kernel and "+policyName)

plt.tight_layout()

plt.savefig('results/dataset_1_'+kernel_dir+'_svr/'+policyName+'/abs_err_diff_'+ S +'_w_'+str(W)+'.png')

'''
Plot histogram of |e-e'|
'''
fig, ax = plt.subplots()

n, bins, patches = ax.hist(err_diff,color='xkcd:azure',bins=(SIZE-100-W)//3, edgecolor='black')

median = np.median(err_diff)

props = dict(boxstyle='round', facecolor='white')
for i in range(1,len(patches)):
    if patches[i-1].xy[0]<=median and median<patches[i].xy[0]:
        patches[i-1].set_color('xkcd:banana')
        patches[i-1].set_edgecolor('black')
        patches[i-1].set_hatch('/')
        ax.text(patches[i-1].xy[0]+patches[i-1].get_width()/2,n[i-1],"median:\n{0:1.6f}".format(median),va='center', color='r', bbox=props)
if patches[i].xy[0]<=median:
    patches[i].set_color('xkcd:banana')
    patches[i].set_edgecolor('black')
    patches[i].set_hatch('/')
    ax.text(patches[i].xy[0]+patches[i].get_width()/2,n[i-1],"median:\n{0:1.6f}".format(median),va='center', color='r', bbox=props)

plt.xlabel("Error rate difference, |e-e'|")
plt.ylabel("Frequency")
plt.title("Absolute error difference for SUV sensor ["+S\
    +"], w="+str(W)+",\nusing Support Vector Regression with "+kernel_name+" Kernel and "+policyName+"\nand the corresponding median for the data")

fig.tight_layout()

plt.savefig('results/dataset_1_'+kernel_dir+'_svr/'+policyName+'/hist_data_dist_median_'+S+'_w_'+str(W)+'.png')

'''
Plot all error rates
'''
fig, ax = plt.subplots()

plt.plot(range(0,len(err_storage)), err_storage[0:], fillstyle='bottom')

ax.hlines(init_err,0,len(err_storage)-1,colors='r')
props = dict(boxstyle='round', facecolor='white')
ax.text(len(err_storage)-0.5,init_err,"{0:f}".format(init_err)\
	,va='center', color='r', bbox=props)
ax.grid(True)
# ax.set_xticks(tuple([0]+list(range(15,len(err_storage)+15,15))))

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel("Window index")
plt.ylabel("Error rate, e")
plt.title("Error rate increase/decrease compared\n"+\
	"to initial error rate for HT sensor system,"+\
	" s="+ S +", w="+str(W)+\
	",\nusing Support Vector Regression with "+kernel_name+" Kernel and "+policyName)

plt.tight_layout()

plt.savefig('results/dataset_1_'+kernel_dir+'_svr/'+policyName+'/err_rates_'+ S +\
	'_w_'+str(W)+'.png')
