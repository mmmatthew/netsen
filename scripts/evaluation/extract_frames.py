"""
These functions make it possible to create multitemporal frames from the videos. The water level information can be provided if available
"""

import pandas
import cv2
import os
import datetime
import glob
import numpy as np


def extract_from_all(video_dir, output_dir, timedeltas, new_dims, waterlevel_data, offsets, force=False):
    # Check that output dir exists
    if os.path.exists(output_dir) and not force:
        return
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all videos downloaded
    for videofile in glob.glob(os.path.join(video_dir, '*.avi')):
        process_video(videofile, timedeltas, new_dims, output_dir, waterlevel_data, offsets, step_s=1)


def process_video(videofilepath, timedeltas, new_dims, data_root, waterlevel_data=None, offsets=None, step_s=1):
    # goes through video and saves a frame every step_s seconds.

    time_step = datetime.timedelta(seconds=step_s)

    # Get video metadata
    session, camera, timedelta, video_start_time, video_end_time = get_video_metadata(videofilepath, offsets)

    # Open video file
    vid_capture = cv2.VideoCapture(videofilepath)

    multitime = os.path.split(data_root)[1]

    # Initialize time
    moment = video_start_time
    video_ms = 0  # the advancement through the video, in ms

    # loop through time steps
    while moment < video_end_time:
        #  First get mean water level
        if waterlevel_data is not None:
            level = waterlevel_data.loc[waterlevel_data['datetime'] == moment]['value'].mean()
            name_image = camera + '_' + multitime + '_' + moment.strftime('%y%m%d_%H%M%S_') + "{:.0f}".format(level) + '.jpg'
        else:
            name_image = camera + '_' + multitime + '_' + moment.strftime('%y%m%d_%H%M%S_') + '.jpg'

        # save image
        image_path = os.path.join(data_root, name_image)
        save_frame(path=image_path, vidcap=vid_capture, vid_time_ms=video_ms, delays=timedeltas, new_dims=new_dims)

        # increment time
        video_ms += step_s * 1000
        moment = moment + time_step


def load_water_level(url, sep=';'):
    # Read the water level data from the file
    waterlevel_data = pandas.read_csv(url, sep=sep, parse_dates=[0], infer_datetime_format=True,
                                      dayfirst=True)
    return waterlevel_data


def load_video_time_offsets(url, sep='\t'):
    # Read temporal offsets from file
    offsets = pandas.read_csv(url, sep=sep, skiprows=2)

    # Reformat data
    offsets2 = pandas.melt(offsets, id_vars=['recording session '], var_name='camera')
    return offsets2


def save_frame(path, vidcap, vid_time_ms, delays, new_dims):
    # saves the water level and video frame to png file
    # extract frames for each delay

    # list of channels
    channels = []

    for d in delays:
        vidcap.set(0, int(vid_time_ms + d * 1000))
        ret, image = vidcap.read()
        if ret:
            # if successful, convert to greyscale
            channels.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        else:
            return 1

    # MERGE Channels
    image = cv2.merge(tuple(channels))

    # downscale image
    image_scaled = cv2.resize(image, new_dims, interpolation=cv2.INTER_CUBIC)

    # Save to file
    filename = path
    cv2.imwrite(filename, image_scaled)


def get_video_metadata(video_filepath, offsets=None):
    video_directory, video_filename = os.path.split(video_filepath)
    # Get temporal offset of video
    info = os.path.split(video_directory)[1].split('_')
    session = info[-1]
    camera = info[1]
    if offsets is not None:
        offset = offsets[(offsets['camera'] == camera+' ') & (offsets['recording session '] == session)].iloc[0]['value']
        if offset[0] == '-':
            isnegative = True
            offset = offset[1:]
        t = datetime.datetime.strptime(offset, "%H:%M:%S")
        timedelta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        if isnegative:
            timedelta = -timedelta
    else:
        timedelta = datetime.timedelta(0)

    # get start and end times of video (Actually, the timestamps in the images should be used, but this is good enough)
    video_info = os.path.splitext(os.path.basename(video_filename))[0].split('_')
    video_start_time = datetime.datetime.strptime(' '.join(video_info[3:5]), "%y%m%d %H%M%S")
    video_end_time = datetime.datetime.strptime(' '.join([video_info[3], video_info[5]]), "%y%m%d %H%M%S")

    # Adjust times to compensate for time shifts!
    video_start_time = video_start_time + timedelta
    video_end_time = video_end_time + timedelta

    # return all info
    return session, camera, timedelta, video_start_time, video_end_time
