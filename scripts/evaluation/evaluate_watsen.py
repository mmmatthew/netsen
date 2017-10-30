import setup
import fetch_videos
import extract_frames
import select_sample_images
import image_datasets
import train_classifier
import segment_frames
import compute_index
import os
import glob
import evaluation_settings as s
import test_classifier
from datetime import datetime

# ### Evaluation settings
# The method is tested with video data from the floodX experiments.

working_dir = 'E:/watson_eval'
# video_archive_url = 'https://zenodo.org/record/830451/files/s3_cam1_instar_161007A.tar'
# video_archive_urls = [
#     'https://zenodo.org/record/830451/files/s3_cam1_instar_161007A.tar',
#     'https://zenodo.org/record/830451/files/s6_cam5_instar_161007A.tar',
#     'https://zenodo.org/record/1035740/files/c4_cam4_instar_161007A.tar',
#     'https://zenodo.org/record/830451/files/c3_cam3_instar_161007A.tar',
#     'https://zenodo.org/record/830451/files/r3_cam2_instar_161007A.tar',
#     ]
# sensor_data_url = 'https://zenodo.org/record/1014028/files/all_s3_h_us_maxbotix.txt'
# sensor_data_urls = [
#     'https://zenodo.org/record/1014028/files/all_s3_h_us_maxbotix.txt',
#     'https://zenodo.org/record/1014028/files/all_s6_h_us_maxbotix.txt',
#     'https://zenodo.org/record/1014028/files/all_c3_h_us_nivus.txt',
#     'https://zenodo.org/record/1014028/files/all_c3_h_us_nivus.txt',
#     'https://zenodo.org/record/1014028/files/all_c3_h_us_nivus.txt']
# camera_time_offset_url = 'https://zenodo.org/record/830451/files/temporal_offsets_of%20cameras.txt'
#
# # ### Set up folder structure
# setup.run(working_dir)
#
#
# # ## Fetch videos
# video_folders = []
# for url in video_archive_urls:
#     video_folders.append(fetch_videos.sync(os.path.join(working_dir, s.stages[0]), url))
#
#
# # ## Extract video frames into multiframe images
# # Get time offset data
# time_offset = extract_frames.load_video_time_offsets(camera_time_offset_url)
#
# for i in range(len(video_archive_urls)):
#     extract_frames.extract_from_all(
#         video_folders[i], os.path.join(working_dir, s.stages[1]),
#         s.frame_extraction_new_dim,
#         sensor_data_urls[i], time_offset)
#
# # ## Select samples randomly
# image_dirs = os.listdir(os.path.join(working_dir, s.stages[1]))
# select_sample_images.create_all(working_dir)

## create datasets
datasets = image_datasets.create_all(
    working_dir=working_dir)

# do training
# for dataset in datasets:
#     train_classifier.train(dataset, working_dir)

# # do testing (INTRA-event performance)
# for model_dir in os.listdir(os.path.join(working_dir, s.stages[4])):
#     dataset = model_dir.split(sep='__')[0] + '.csv'
#     test_classifier.test(
#         model_dir=os.path.join(working_dir, s.stages[4], model_dir),
#         working_dir=working_dir, dataset_csv=os.path.join(working_dir, s.stages[3], dataset), force=False
#     )
# for prediction_dir in os.listdir(os.path.join(working_dir, s.stages[5])):
#     dataset_path = os.path.join(working_dir, s.stages[3], prediction_dir.split('__')[0] + '.csv')
#     iou = test_classifier.computeIou(dataset_path, os.path.join(working_dir, s.stages[5], prediction_dir))
#     print(prediction_dir, iou)

# # do testing (INTER-event performance)
# for model_dir in os.listdir(os.path.join(working_dir, s.stages[4])):
#     # get multitime
#     multitime = model_dir.split(sep='__')[0].split('_', maxsplit=2)[-1]
#     datasets = glob.glob(os.path.join(working_dir, s.stages[3], '*' + multitime + '.csv'))
#     for dataset in datasets:
#         test_classifier.test(
#             model_dir=os.path.join(working_dir, s.stages[4], model_dir),
#             working_dir=working_dir, dataset_csv=dataset, force=False
#         )
# test_results = []
# for prediction_dir in os.listdir(os.path.join(working_dir, s.stages[5])):
#     if os.path.isdir(os.path.join(working_dir, s.stages[5], prediction_dir)):
#         dataset_path = os.path.join(working_dir, s.stages[3], prediction_dir.split('__D')[1] + '.csv')
#         iou = test_classifier.computeIou(dataset_path, os.path.join(working_dir, s.stages[5], prediction_dir), channel=2)
#         test_results.append(prediction_dir + " " + str(iou))
#
# with open(os.path.join(working_dir, s.stages[5], 'test_results_' + datetime.now().strftime('%Y-%m-%d %H%M%S') + '.txt'), 'w') as f:
#     for item in test_results:
#         f.write("%s\n" % item)

# # Predict complete time series
# for sequence_dir in os.listdir(os.path.join(working_dir, s.stages[1])):
#     # Run segmentation
#     segment_frames.run(os.path.join(working_dir, s.stages[1], sequence_dir), working_dir)

# evaluate sequence
for images_dir in os.listdir(os.path.join(working_dir, s.stages[6])):
    compute_index.process_images(directory=os.path.join(working_dir, s.stages[6], images_dir), working_directory=working_dir)

for ts in glob.glob(os.path.join(working_dir, s.stages[7], '*.csv')):
    compute_index.plot_from_csv(ts, os.path.join(working_dir, s.stages[7]))
