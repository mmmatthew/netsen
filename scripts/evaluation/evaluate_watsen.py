
# coding: utf-8

# # Evaluation of WATSEN
# ## Some housekeeping
# 

# In[17]:

# For reloading external modules
# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')


# In[67]:

import setup
import fetch_videos
import extract_frames
import select_sample_images
import json
import os


# ### Evaluation settings
# The method is tested with video data from the floodX experiments.
# 

# In[68]:

working_dir='E:/watson_eval'
video_archive_url='https://zenodo.org/record/830451/files/s3_cam1_instar_161007A.tar'
water_level_data_url='https://zenodo.org/record/1014028/files/all_s3_h_us_maxbotix.txt'
camera_time_offset_url='https://zenodo.org/record/830451/files/temporal_offsets_of%20cameras.txt'

# read more evaluation settings
with open('evaluation_settings.json') as json_data:
    settings = json.load(json_data)


# ### Set up folder structure
# 

# In[20]:

stages = setup.run(working_dir)


# ## Fetch videos

# In[21]:

video_folder = fetch_videos.sync(stages[0], video_archive_url)


# ## Extract video frames into multiframe images
# 

# In[54]:

# Get water level and time offset data
water_levels = extract_frames.load_water_level(water_level_data_url)
time_offset = extract_frames.load_video_time_offsets(camera_time_offset_url)


# In[66]:

image_dirs = []
for timedeltas in settings['frame_extraction_combinations']:
    output_dir = os.path.join(stages[1], '_'.join(str(x) for x in timedeltas))
    extract_frames.extract_from_all(
        video_folder, output_dir, timedeltas, 
        tuple(settings['frame_extraction_new_dim']), 
        water_levels, time_offset)
    image_dirs.append(output_dir)


# ## Select samples randomly
# 

# In[78]:

select_sample_images.create(image_dir=image_dirs[3], output_dir=stages[2], settings=settings)

