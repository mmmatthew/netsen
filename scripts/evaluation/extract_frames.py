"""
These functions make it possible to create multitemporal frames from the videos. The water level information can be provided if available
"""

import pandas
import cv2
import os
import datetime
import glob
import evaluation_settings as s


def extract_from_all(video_dir, output_dir, new_dims, sensor_data_url, offsets, force=False):

    # Get sensor data
    sensor_data = load_sensor_data(sensor_data_url)

    # Depending on the camera, we either do many or few combinations
    if 'cam1' in video_dir:
        # use all combinations
        combinations = s.frame_extraction_combinations_large
    else:
        combinations = s.frame_extraction_combinations_small

    # Extracts frames from all videos for a given camera, for all different timedeltas
    for timedeltas in combinations:
        # create multitime string
        multitime = '_'.join(str(x) for x in timedeltas)
        # get camera from directory
        camera = os.path.basename(video_dir).split('_')[1]
        # subdir for saving the frames
        output_subdir = os.path.join(output_dir, camera + '_' + multitime)

        # Check that output dir exists and skip if so
        if os.path.exists(output_subdir) and not force:
            continue
        elif not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        # Process all videos downloaded
        for videofile in glob.glob(os.path.join(video_dir, '*.avi')):
            process_video(videofile, timedeltas, new_dims, output_subdir, sensor_data, offsets, step_s=1, force=force)


def process_video(videofilepath, timedeltas, new_dims, output_subdir, sensor_data=None, offsets=None, step_s=1, force=False):
    # create multitime string
    multitime = '_'.join(str(x) for x in timedeltas)

    # goes through video and saves a frame every step_s seconds.
    time_step = datetime.timedelta(seconds=step_s)

    # Get video metadata
    session, camera, timedelta, video_start_time, video_end_time = get_video_metadata(videofilepath, offsets)

    # Open video file
    vid_capture = cv2.VideoCapture(videofilepath)

    # Initialize time
    moment = video_start_time
    video_ms = 0  # the advancement through the video, in ms

    # loop through time steps
    while moment < video_end_time:

        level = int(sensor_data.loc[moment]['value'])
        name_image = camera + '_' + multitime + '_' + moment.strftime('%y%m%d_%H%M%S_') + "{:.0f}".format(level) + '.jpg'


        # save image
        image_path = os.path.join(output_subdir, name_image)
        save_frame(path=image_path, vidcap=vid_capture, vid_time_ms=video_ms, delays=timedeltas, new_dims=new_dims)

        # increment time
        video_ms += step_s * 1000
        moment = moment + time_step


def load_sensor_data(url, sep=';'):
    # Read the water level data from the file, resampled to a second frequency

    waterlevel_data = pandas.read_csv(url, sep=sep, parse_dates=[0], infer_datetime_format=True,
                                      dayfirst=True, index_col=0)
    # Resample to second frequency
    if any(waterlevel_data.index.duplicated()):
        # this is fast data: downsample
        resampled = waterlevel_data.resample('S').mean()
    else:
        upsampled = waterlevel_data.resample('S')
        resampled = upsampled.interpolate(method='linear')
    return resampled


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
    try:
        image_scaled = cv2.resize(image, new_dims, interpolation=cv2.INTER_CUBIC)
    except TypeError:
        print('something happened')
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
        else:
            isnegative = False
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
