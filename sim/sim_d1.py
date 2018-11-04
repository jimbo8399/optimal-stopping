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
from bin.data_import import import_dataset_1 as im
from svr.svr_model import k_fold_cv as get_error

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

# Initialising data structure
data_init()
sensor_dataset = im().iloc[0:60,:]

dataset_length = len(sensor_dataset)

if dataset_length<W:
	print("insufficient amount of data")
	exit(1)

#print("Epoch 0 ,window",0, 0+W)
data = sensor_dataset.iloc[0:W,:]

# Reshape the temperature and humidity values
r_th1 = data[['Temp.','Humidity']].values
# Reshape the sensor values
r_s1 = data[[S]].values
# Build a model to be sent to the Edge Gate
model = get_model(r_th1, r_s1)
# Evaluate the model
err = get_error(model, r_th1, r_s1)

err_diff = []
err_storage = [err]
init_err = err

i = 1
while (i + W) <= dataset_length:
	# Receive a new datapoint
	# print()
	# print("Epoch",i,",window",i, i+W)
	data = sensor_dataset.iloc[i:i+W,:]
	r_th2 = data[['Temp.','Humidity']].values
	r_s2 = data[[S]].values
	# Build a new model with the newly arrived datapoint 
	# and the discarded oldest datapoint
	new_model = get_model(r_th2, r_s2)
	# Evaluate
	new_err = get_error(new_model, r_th1, r_s1)
	err_storage += [new_err]

	err_diff += [abs(err-new_err)]

	# print(err, new_err)
	# print(err_diff[-1])
	# Slide the window with 1
	i += 1


'''
Plot Error rate difference
'''
fig, ax = plt.subplots()
ax.grid(True)
ax.set_xticks(tuple(range(1,len(err_storage)+1,2)))

plt.plot(range(1,len(err_diff)+1), err_diff, fillstyle='bottom')

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.xlabel("Window index")
plt.ylabel("Error rate difference, |e-e'|")
plt.title("Absolute error difference for HT sensor system,"+ \
	" s="+ S +", w="+str(W)+\
	",\nusing Support Vector Regression with "+kernel_name+" Kernel")

plt.tight_layout()

plt.savefig('results/dataset_1_'+kernel_dir+'_svr/abs_err_diff_'+ S +'_w_'+str(W)+'.png')

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
ax.set_xticks(range(0,len(err_storage)+1,2))

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel("Window index")
plt.ylabel("Error rate, e")
plt.title("Error rate increase/decrease compared\n"+\
	"to initial error rate for HT sensor system,"+\
	" s="+ S +", w="+str(W)+\
	",\nusing Support Vector Regression with "+kernel_name+" Kernel")

plt.tight_layout()

plt.savefig('results/dataset_1_'+kernel_dir+'_svr/err_rates_'+ S +\
	'_w_'+str(W)+'.png')