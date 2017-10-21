import setup
import fetch_videos
import extract_frames
import select_sample_images
import image_datasets
import json
import os


# ### Evaluation settings
# The method is tested with video data from the floodX experiments.

working_dir='E:/watson_eval'
video_archive_url='https://zenodo.org/record/830451/files/s3_cam1_instar_161007A.tar'
water_level_data_url='https://zenodo.org/record/1014028/files/all_s3_h_us_maxbotix.txt'
camera_time_offset_url='https://zenodo.org/record/830451/files/temporal_offsets_of%20cameras.txt'

# read more evaluation settings
with open('evaluation_settings.json') as json_data:
    settings = json.load(json_data)

# ### Set up folder structure
stages = setup.run(working_dir)


# ## Fetch videos
video_folder = fetch_videos.sync(stages[0], video_archive_url)


# ## Extract video frames into multiframe images
# Get water level and time offset data
water_levels = extract_frames.load_water_level(water_level_data_url)
time_offset = extract_frames.load_video_time_offsets(camera_time_offset_url)

image_dirs = []
for timedeltas in settings['frame_extraction_combinations']:
    output_dir = os.path.join(stages[1], '_'.join(str(x) for x in timedeltas))
    extract_frames.extract_from_all(
        video_folder, output_dir, timedeltas, 
        tuple(settings['frame_extraction_new_dim']), 
        water_levels, time_offset)
    image_dirs.append(output_dir)

# ## Select samples randomly
images_path, labels_path = select_sample_images.create(image_dir=image_dirs[3], output_dir=stages[2], settings=settings)

# ## create datasets
image_datasets.create(
    label_dir=labels_path['intra-event'],
    image_dir=images_path['intra-event'],
    output_dir=stages[3],

)