import pandas as pd
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

#function to return an array of data file lengths
def find_sequence_lengths(data_dir):
    satellite_data = Path(data_dir).glob('*.csv') #load directory containing all data
    sequence_lengths = {} # dictionary to store file and file length key value pairs
    if not satellite_data:
        raise ValueError(f'No csv files found in {data_dir}')
    for data_file in satellite_data:
        data_df = pd.read_csv(data_file)
        sequence_lengths[int(data_file.stem)] = len(data_df)
    
    return sequence_lengths

# function to create data sequences
def create_sequences(data,interval):
    sequences = [] #array to hold data sequences
    end_idx = 0
    start_idx = 0
    while end_idx < len(data):
        end_idx = start_idx + interval
        # Ensure the end index does not exceed the length of the data
        if end_idx > len(data):
            end_idx = len(data)
        # Append the sequence if it's not empty
        if start_idx < end_idx:
            sequences.append(data[start_idx:end_idx])
        start_idx = end_idx

    max_rows = max(seq.shape[0] for seq in sequences)
    if len(data.shape) > 1:
        return np.array([np.pad(seq, ((0, max_rows - seq.shape[0]), (0, 0)), mode='constant') for seq in sequences])
    else:
        return np.array([np.pad(seq, (0, max_rows - seq.shape[0]), mode='constant') for seq in sequences])
    #return np.array(sequences,dtype='object')

# function to detect outliers in the dataset usiing IQR method
def detect_outliers_IQR(data,features):
    outlier_result = {} #output decitionary containing key value pairs representing feature name 
                        #and number of outliers respectively
    for feature in features:
        Q1 = data[feature].quantile(0.25) #first quantile of the feature
        Q3 = data[feature].quantile(0.75) #third quantile of the feature
        IQR = Q3-Q1 #calculation of inter quartile distance
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR

        #Identifying outlier
        num_outliers = len(data[(data[feature] < lower_bound) | (data[feature] > upper_bound)])
        outlier_result[feature] = num_outliers

    return outlier_result