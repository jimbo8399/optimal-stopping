import pandas as pd
import json

#TODO: fix paths
PATH_DATASET_1 = "./data/DATASET-1/HT_Sensor_dataset.dat"
PATH_DATASET_2 = "./data/DATASET-2/pi"

'''
Imports a single sensor dataset from GNFUV
Returns: pandas dataframe
'''
def import_sensor_from_dataset_2(path):
    with open(path,"r") as f:
        line_of_data = next(f)
        df = pd.read_json(json.dumps(line_of_data),lines=True)
        for line_of_data in f:
            temp_df = pd.read_json(json.dumps(line_of_data), lines=True)
            df = df.append(temp_df)
    return df

'''
Imports the whole dataset from
'''
def import_dataset_2():
    df_pi2 = import_sensor_from_dataset_2(PATH_DATASET_2 + 
                         "2/gnfuv-temp-exp1-55d487b85b-5g2xh_1.0.csv")
    df_pi3 = import_sensor_from_dataset_2(PATH_DATASET_2 +
                         "3/gnfuv-temp-exp1-55d487b85b-2bl8b_1.0.csv")
    df_pi4 = import_sensor_from_dataset_2(PATH_DATASET_2 +
                         "4/gnfuv-temp-exp1-55d487b85b-xcl97_1.0.csv")
    df_pi5 = import_sensor_from_dataset_2(PATH_DATASET_2 +
                         "5/gnfuv-temp-exp1-55d487b85b-5ztk8_1.0.csv")
    return [df_pi2, df_pi3, df_pi4, df_pi5]

def import_dataset_1():
    with open(PATH_DATASET_1, "r") as file:
        columns = next(file).split()
        df = pd.read_csv(file,names=columns,delim_whitespace=True)
    return df

# if __name__=="__main__":
#     import_dataset_1()