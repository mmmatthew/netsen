"""
These functions extract sample images randomly from predefined periods to be labeled.
"""
import os
import pandas
import glob
import shutil
import evaluation_settings as s
import random
import numpy as np

random.seed(1)


def create_all(working_dir):
    """
    Goes through working dir and selects frames to sample images from to be labeled.
    :param working_dir: Where to look for directories of extracted frames
    :return: None.
    """
    for key, sample_settings in s.select_sample_images.items():
        for cam in sample_settings['cameras']:
            # todo: get a image folder for the given camera
            image_folders = glob.glob(os.path.join(working_dir, s.stages[1], cam + '*'))
            if len(image_folders) > 0:
                # take images from the first folder for sampling. All other folder should have equivalent files.
                create(image_dir=image_folders[0], output_dir=os.path.join(working_dir, s.stages[2]), camera=cam, sample_settings=sample_settings)
            else:
                print('No images found for ', cam)


def get_weights(file_list, weight_flooded, threshold=5):
    if weight_flooded == 1:
        return None
    else:
        w = [weight_flooded if float(os.path.splitext(f)[0].split('_')[-1]) > threshold else 1 for f in file_list]
        s = sum(w)
        return [e/s for e in w]


def create(image_dir, output_dir, camera, sample_settings, image_pattern='*.jpg', force=False):
    """
    Selects frames randomly for labelling
    :param image_dir: Where the frames are located.
    :param output_dir: Where the selection should be saved to.
    :param camera: Camera name, for naming new directory
    :param sample_settings: Settings for random sampling
    :param image_pattern: Regular expression for finding frames in image_dir
    :param force: Whether samples should be re-generated even if they already exist.
    :return: Path to images and labels
    """
    # set up structure
    dir_name = camera + '_' + sample_settings['name']
    images_path, labels_path = directory_struct(output_dir, dir_name, force)

    image_list = glob.glob(os.path.join(image_dir, image_pattern))

    # check if there are already files in the folders
    if len(os.listdir(images_path)) > 0 and not force:
        print(dir_name + ' already exists. use force=True to overwrite')
    else:
        if force:
            # delete existing files
            for the_file in os.listdir(images_path):
                file_path = os.path.join(images_path, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

        # only select images within date
        filtered_list = filter_by_date(
            image_list,
            sample_settings['start'],
            sample_settings['end'])

        weights = get_weights(filtered_list, sample_settings['encourage_flooded'])

        rand_selection = np.random.choice(a=filtered_list, size=sample_settings['count'], p=weights)

        # write to file
        # with open(os.path.join(output_dir, dir_name+'.txt'), 'w') as file:
        #     file.writelines(["%s\n" % item for item in rand_selection])

        # copy files
        for path in rand_selection:
            shutil.copy(path, images_path)
    return images_path, labels_path

def directory_struct(directory, name, force):

    subdirs = [
        'images',
        'labels'
    ]
    if not os.path.exists(os.path.join(directory, name)):
        os.makedirs(os.path.join(directory, name))

    for dir in subdirs:
        if not os.path.exists(os.path.join(directory, name, dir)):
            os.makedirs(os.path.join(directory, name, dir))
    images_path = os.path.join(directory, name, 'images')
    labels_path = os.path.join(directory, name, 'labels')
    return images_path, labels_path


def filter_by_date(path_list, start, end):
    images = pandas.DataFrame({'path': path_list})
    # time info is right before the water level
    images['time'] = ['_'.join(os.path.basename(os.path.splitext(p)[0]).split('_')[-3:-1]) for p in images['path']]
    # noinspection PyChainedComparisons
    filtered = images.loc[(start < images['time']) & (images['time'] <= end), 'path']

    return list(filtered)
