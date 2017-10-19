"""
These functions extract sample images randomly from predefined periods to be labeled.
"""
import os
import pandas
import glob
import random
import shutil


def create(image_dir, output_dir, settings, image_pattern='*.jpg', prepend='', force=False):
    # set up structure
    images_path, labels_path = directory_struct(output_dir, force)

    image_list = glob.glob(os.path.join(image_dir, image_pattern))

    for key, dest_path in images_path.items():

        # check if there are already files in the folders
        if len(os.listdir(dest_path)) > 0 and not force:
            print(key + ' already exists. use force=True to overwrite')
        else:
            if force:
                for the_file in os.listdir(dest_path):
                    file_path = os.path.join(dest_path, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)
            filtered_list = filter_by_date(
                image_list,
                settings['select_sample_images'][key]['start'],
                settings['select_sample_images'][key]['end'])

            rand_selection = random.sample(filtered_list, settings['select_sample_images'][key]['count'])

            # write to file
            with open(os.path.join(output_dir, key+'.txt'), 'w') as file:
                file.writelines(["%s\n" % item for item in rand_selection])

            # copy files
            for path in rand_selection:
                shutil.copy(path, dest_path)
    return labels_path

def directory_struct(directory, force):
    subdirs = [
        'intra-event',
        'inter-event'
    ]
    subsubdirs = [
        'images',
        'labels'
    ]
    images_path = {}
    labels_path = {}
    for subdir in subdirs:
        if not os.path.exists(os.path.join(directory, subdir)):
            os.makedirs(os.path.join(directory, subdir))

        for ssdir in subsubdirs:
            if not os.path.exists(os.path.join(directory, subdir, ssdir)):
                os.makedirs(os.path.join(directory, subdir, ssdir))
        images_path[subdir] = os.path.join(directory, subdir, 'images')
        labels_path[subdir] = os.path.join(directory, subdir, 'labels')
    return images_path, labels_path


def filter_by_date(path_list, start, end):
    images = pandas.DataFrame({'path': path_list})
    # time info is right before the water level
    images['time'] = ['_'.join(os.path.basename(os.path.splitext(p)[0]).split('_')[-3:-1]) for p in images['path']]
    # noinspection PyChainedComparisons
    filtered = images.loc[(start < images['time']) & (images['time'] <= end), 'path']

    return list(filtered)
