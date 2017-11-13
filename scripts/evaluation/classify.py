


import pandas
import os
import evaluation_settings as s
import glob
from datetime import datetime as dt
import matplotlib.pyplot as plt

def process_all(working_dir):
    output_dir = os.path.join(working_dir, s.stages[8])
    for csv in glob.glob(os.path.join(working_dir, s.stages[7], '*cam1_0_0.2_0.4.csv')):
        # check if already exists
        if os.path.exists(os.path.join(output_dir, os.path.basename(csv))):
            continue
        else:
            classify(csv, output_dir)


def classify(csv, output_dir):
    data = pandas.read_csv(csv)
    # smooth data
    data['flood_index_smoothed'] = data.flood_index.rolling(win_type='gaussian', window=60, center=True).mean(std=30)
    data['sensor_value_smoothed'] = data.sensor_value.rolling(win_type='gaussian', window=60, center=True).mean(std=30)
    # parse dates and set as index
    data['datetime'] = [dt.strptime(d, '%y%m%d_%H%M%S') for d in data.datetime]
    data = data.set_index('datetime')
    # diff data
    data_change = data.diff()
    # merge the two
    data_merged = pandas.merge(data, data_change, left_index=True, right_index=True, suffixes=('', '_diff'))


def plot(data):
    plt.plot(data.flood_index)
    plt.plot(data.flood_index_smoothed)
    plt.plot(data.sensor_value_smoothed/500)