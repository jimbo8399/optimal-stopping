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
from linreg.lin_reg_model import get_linear_regression_model as get_model
from linreg.lin_reg_model import k_fold_cv as get_error

W = 10 # window size

# Initialising data structure
data_init()
# Import data from Dataset 2
sensor = im()[0].iloc[:30,:]

dataset_length = len(sensor)

if dataset_length<10:
	print("insufficient amount of data")
	exit(1)

print("Epoch 0 ,window",0, 0+W)
data = sensor.iloc[0:W,:]
# Reshape the temperature values
r_t1 = data.temperature.values.reshape(-1,1)
# Reshape the humidity values
r_h1 = data.humidity.values.reshape(-1,1)
# print(data.iloc[:,2:4]) # DEBUG
# Build a model to be sent to the Edge Gate
model = get_model(r_t1, r_h1)
# Evaluate the model
err = get_error(model, r_t1, r_h1)

i = 1
while (i + W) <= dataset_length:
	# Receive a new datapoint
	print()
	print("Epoch",i,",window",i, i+W)
	data = sensor.iloc[i:i+W,:]
	# print(data.iloc[:,2:4]) # DEBUG
	r_t2 = data.temperature.values.reshape(-1,1)
	r_h2 = data.humidity.values.reshape(-1,1)
	# Build a new model with the newly arrived datapoint 
	# and the discarded oldest datapoint
	new_model = get_model(r_t2, r_h2)
	# Evaluate
	new_err = get_error(new_model, r_t1, r_h2)

	print(err, new_err)
	# Slide the window with 1
	i += 1