"""
these functions read a directory of segmented images and computes the qualitative flood index for each.
The following options can be used: definition of a region of interest, thresholds.
The result is saved as a csv which includes the flood indexes, the time and date, and water level
"""

import evaluation_settings as s
import os
from glob import glob
import pandas
from PIL import Image
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as md


def process_images(directory, working_directory, image_pattern='*.png'):
    output_dir = os.path.join(working_directory, s.stages[7])
    datetime, sensor_value, flood_index = [], [], []
    camera = os.path.basename(directory).split('__')[-1].split('_')[0]
    output_filepath = os.path.join(output_dir, os.path.basename(directory) + '.csv')
    if camera in s.rois.keys() and not os.path.exists(output_filepath):
        for image_path in glob(os.path.join(directory, image_pattern)):
            t, sv, f = compute_index(image_path)
            datetime.append(t)
            sensor_value.append(sv)
            flood_index.append(f)

        data = pandas.DataFrame(dict(datetime=datetime, sensor_value=sensor_value, flood_index=flood_index))

        data.to_csv(output_filepath)




def compute_index(image_path, channel=2, threshold=0.9):
    # get info from filename
    date, time, sensor_value = os.path.splitext(image_path)[0].split('_')[-3:]
    camera = os.path.basename(image_path).split('_')[0]
    if camera in s.rois.keys():
        # read image channel as numpy array
        flood_array = (np.array(Image.open(image_path), dtype=np.float32)/s.y_scaling)[..., channel].squeeze()
        # crop to region of interest
        top = s.rois[camera]['top']
        left = s.rois[camera]['left']
        height = s.rois[camera]['height']
        width = s.rois[camera]['width']
        flood_array_cropped = flood_array[top:(top+height), left:(left+width)]
        # compute index
        index = (flood_array_cropped > threshold).sum()/(flood_array_cropped.shape[0] * flood_array_cropped.shape[1])
        # return datetime, sensor value, and flood index
        return '_'.join([date, time]), sensor_value, index


def plot_ts_matplotlib(data, save_path):
    fig, axarr = plt.subplots(2, sharex=True, figsize=(12, 5))
    p1 = axarr[0].plot(data['datetime'], data['flood_index'].rolling(30, center=True).mean(), 'r', label='flood index')

    p2 = axarr[1].plot(data['datetime'], data['sensor_value'], label='real water level')
    # p3 = ax2.scatter(training['time'], training['sensor'], marker='x', s=40, c='black', label='training data')

    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    axarr[0].legend()
    axarr[1].legend()

    plt.xlabel('time')
    axarr[1].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    axarr[0].set_ylabel('WATSEN prediction')
    axarr[0].yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    axarr[1].set_ylabel('Measured water level')
    axarr[1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{0:g} m'.format(x / 1000)))

    plt.savefig(save_path)

def plot_scatter_matplotlib(data, save_path):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(data['sensor_value'], data['flood_index'], marker='.', markersize=1, linestyle='None')
    ax.set_ylabel('WATSEN prediction')
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    ax.set_xlabel('Measured water level')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{0:g} m'.format(x / 1000)))

    plt.savefig(save_path)

def plot_from_csv(csv_path, save_path):
    data = pandas.read_csv(csv_path)
    # parse time
    data['datetime'] = [dt.strptime(d, '%y%m%d_%H%M%S') for d in data.datetime]
    plot_ts_matplotlib(data, save_path=os.path.join(save_path, os.path.splitext(os.path.basename(csv_path))[0] + '_ts.png'))
    plot_scatter_matplotlib(data, save_path=os.path.join(save_path, os.path.splitext(os.path.basename(csv_path))[0] + '_scatter.png'))