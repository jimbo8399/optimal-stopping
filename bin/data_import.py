import pandas as pd
import json
from pathlib import Path

proj_path = Path("/home/katya/optimal-stopping")
data_path = proj_path / 'data'

PATH_DATASET_1 = data_path / "DATASET-1/HT_Sensor_dataset.dat"
PATH_DATASET_2 = data_path / "DATASET-2/pi"

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
def import_dataset_2():
    df_pi2 = import_sensor_from_dataset_2(str(PATH_DATASET_2) + 
                         "2/gnfuv-temp-exp1-55d487b85b-5g2xh_1.0.csv")
    df_pi3 = import_sensor_from_dataset_2(str(PATH_DATASET_2) +
                         "3/gnfuv-temp-exp1-55d487b85b-2bl8b_1.0.csv")
    df_pi4 = import_sensor_from_dataset_2(str(PATH_DATASET_2) +
                         "4/gnfuv-temp-exp1-55d487b85b-xcl97_1.0.csv")
    df_pi5 = import_sensor_from_dataset_2(str(PATH_DATASET_2) +
                         "5/gnfuv-temp-exp1-55d487b85b-5ztk8_1.0.csv")
    return [df_pi2, df_pi3, df_pi4, df_pi5]

'''
Import the whole dataset from the HT_Sensors
A single dataset with the columns:
[id, time, R1, R2, R3, R4, R5, R6, R7, R8, temperature, humidity]
Return: pandas dataframe
'''
def import_dataset_1():
    with open(PATH_DATASET_1, "r") as file:
        columns = next(file).split()
        df = pd.read_csv(file,names=columns,delim_whitespace=True)
    return df

# if __name__=="__main__":
#     import_dataset_2()