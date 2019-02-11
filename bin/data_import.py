import pandas as pd
import json
from pathlib import Path
import os
from urllib import request
from io import BytesIO
from zipfile import ZipFile,ZipInfo

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

# Create path to the project data
data_path = proj_path / 'data'

# Create path to the project results
results_path = proj_path / 'results'
results_raw_path = results_path / 'raw_data'

d1_filename = "HT_Sensor_dataset.dat"
d1_zip_filename = "HT_Sensor_dataset.zip"

PATH_DATASET_1 = data_path / "DATASET-1"
PATH_DATASET_2 = data_path / "DATASET-2"

zip_d1_path = PATH_DATASET_1/"zip_d1.zip"
zip_d2_path = PATH_DATASET_2/"zip_d2.zip"

D1_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00362/HT_Sensor_UCIsubmission.zip"
D2_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00452/GNFUV%20USV%20Dataset.zip"

'''
Create data file structure
Download the needed data
'''
def data_init():
    POLICIES = ['policyE','policyN', 'policyM', 'policyA', 'policyC', 'policyR', 'policyOST']
    '''Data'''
    if "data" not in os.listdir(proj_path):
        print("Creating top directory data/")
        os.mkdir(data_path)
    '''Results'''
    if "results" not in os.listdir(proj_path):
        print("Creating top directory results/")
        os.mkdir(results_path)
    '''Results pickle files'''
    if "raw_data" not in os.listdir(results_path):
        print("Creating directory results/raw_data")
        os.mkdir(results_raw_path)
    '''Results rbf svr'''
    if "dataset_1_rbf_svr" not in os.listdir(results_path):
        print("Creating directory results/dataset_1_rbf_svr")
        os.mkdir(results_path/"dataset_1_rbf_svr")
    for pol in POLICIES:
        if pol not in os.listdir(results_path/"dataset_1_rbf_svr"):
            print("Creating directory results/dataset_1_rbf_svr/"+pol)
            os.mkdir(results_path/"dataset_1_rbf_svr"/pol)
    '''Results lin svr'''
    if "dataset_1_lin_svr" not in os.listdir(results_path):
        print("Creating directory results/dataset_1_lin_svr")
        os.mkdir(results_path/"dataset_1_lin_svr")
    for pol in POLICIES:
        if pol not in os.listdir(results_path/"dataset_1_lin_svr"):
            print("Creating directory results/dataset_1_lin_svr/"+pol)
            os.mkdir(results_path/"dataset_1_lin_svr"/pol)
    '''Results lin reg'''
    if "dataset_2_lin_reg" not in os.listdir(results_path):
        print("Creating directory results/dataset_2_lin_reg")
        os.mkdir(results_path/"dataset_2_lin_reg")
    for pol in POLICIES:
        if pol not in os.listdir(results_path/"dataset_2_lin_reg"):
            print("Creating directory results/dataset_2_lin_reg/"+pol)
            os.mkdir(results_path/"dataset_2_lin_reg"/pol)

    '''
    Fetching and Extracting DATASET 1 - HT Sensors
    '''
    if "DATASET-1" not in os.listdir(data_path):
        print("-->Creating directory data/DATASET-1")
        os.mkdir(PATH_DATASET_1)

    if "zip_d1.zip" not in os.listdir(PATH_DATASET_1) and \
        d1_filename not in os.listdir(PATH_DATASET_1):
        print("-->-->Downloading data from HT Sensors inside data/DATASET-1")
        resp = request.urlopen(D1_URL)
        with open(zip_d1_path,"wb") as file:
            file.write(resp.read())
        print("-->-->Unzipping data")
        zip_file = ZipFile(str(zip_d1_path))
        zip_file.extract(member=d1_zip_filename, path=str(PATH_DATASET_1))
        zip_file = ZipFile(str(PATH_DATASET_1/d1_zip_filename))
        zip_file.extract(member=d1_filename, path=str(PATH_DATASET_1))
    elif d1_filename not in os.listdir(PATH_DATASET_1):
        print("-->-->Unzipping data from zip_d1.zip")
        zip_file = ZipFile(str(zip_d1_path))
        zip_file.extract(member=d1_zip_filename, path=str(PATH_DATASET_1))
        zip_file = ZipFile(str(PATH_DATASET_1/d1_zip_filename))
        zip_file.extract(member=d1_filename, path=str(PATH_DATASET_1))

    '''
    Fetching and extracting Dataset 2 - USV Sensors
    '''
    if "DATASET-2" not in os.listdir(data_path):
        print("-->Creating directory data/DATASET-2")
        os.mkdir(PATH_DATASET_2)

    if "zip_d2.zip" not in os.listdir(PATH_DATASET_2) and \
        ("pi2" not in os.listdir(PATH_DATASET_2) or \
        "pi3" not in os.listdir(PATH_DATASET_2) or \
        "pi4" not in os.listdir(PATH_DATASET_2) or \
        "pi5" not in os.listdir(PATH_DATASET_2)):

        print("-->-->Downloading data from USV Sesnors inside data/DATASET-2")
        resp = request.urlopen(D2_URL)
        with open(zip_d2_path,"wb") as file:
            file.write(resp.read())

        print("-->-->Unzipping data")
        zip_file = ZipFile(str(zip_d2_path))
        zip_file.extractall(path=str(PATH_DATASET_2))
    elif "pi2" not in os.listdir(PATH_DATASET_2) or \
        "pi3" not in os.listdir(PATH_DATASET_2) or \
        "pi4" not in os.listdir(PATH_DATASET_2) or \
        "pi5" not in os.listdir(PATH_DATASET_2):

        print("-->-->Unzipping data from zip_d2.zip")
        zip_file = ZipFile(str(zip_d2_path))
        zip_file.extractall(path=str(PATH_DATASET_2))

'''
Imports a single sensor dataset from GNFUV
A dataset from1 source with the columns:
[id, humidity, temperature, experiment_id, time]
Returns: pandas dataframe
'''
def import_sensor_from_dataset_2(path):
    with open(path,"r") as f:
        line_of_data = next(f).strip()
        line_of_data = line_of_data.replace("'",'"')
        df = pd.read_json(line_of_data,lines=True,orient="records")
        for line_of_data in f:
            line_of_data = line_of_data.strip()
            line_of_data = line_of_data.replace("'",'"')
            try:
                temp_df = pd.read_json(line_of_data, lines=True, orient="records")
            except ValueError:
                continue
            df = df.append(temp_df)
    return df

'''
Imports the whole dataset from GNFUV
Data contains datasets from 4 different sources
Returns: a list with all pandas dataframes
'''
def import_dataset_2(sensors):
    dfs = []
    if "pi2" in sensors:
        dfs += [import_sensor_from_dataset_2(str(PATH_DATASET_2 / 
                             "pi2"/"gnfuv-temp-exp1-55d487b85b-5g2xh_1.0.csv"))]
    if "pi3" in sensors:
        dfs += [import_sensor_from_dataset_2(str(PATH_DATASET_2 /
                             "pi3"/"gnfuv-temp-exp1-55d487b85b-2bl8b_1.0.csv"))]
    if "pi4" in sensors:
        dfs += [import_sensor_from_dataset_2(str(PATH_DATASET_2 /
                             "pi4"/"gnfuv-temp-exp1-55d487b85b-xcl97_1.0.csv"))]
    if "pi5" in sensors:
        dfs += [import_sensor_from_dataset_2(str(PATH_DATASET_2 /
                             "pi5"/"gnfuv-temp-exp1-55d487b85b-5ztk8_1.0.csv"))]
    return dfs

'''
Import the whole dataset from the HT_Sensors
A single dataset with the columns:
[id, time, R1, R2, R3, R4, R5, R6, R7, R8, temperature, humidity]
Return: pandas dataframe
'''
def import_dataset_1():
    with open(PATH_DATASET_1/"HT_Sensor_dataset.dat", "r") as file:
        columns = next(file).split()
        df = pd.read_csv(file,names=columns,delim_whitespace=True)
    return df

# if __name__=="__main__":
#     data_init(proj_path)
