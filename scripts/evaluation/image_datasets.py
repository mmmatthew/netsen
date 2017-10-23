"""
Given directories of labeled images, divide into sets for training, testing, and validation.
Options: natural data augmentation, train/test/validation fractions, reduction in training size
"""

import pandas
import os
from glob import glob


def create(label_dir, image_dir, output_dir, name, augment=False, augment_duration=0,
           frac_train=0.7, frac_test=0.3, frac_validate=0.0, reduce_train=0.0,
           label_pattern='*.png', image_ext='.jpg'):

    # fetch list of label images
    labels = glob(os.path.join(label_dir, label_pattern))
    labels.sort()
    labels_df = pandas.DataFrame(dict(label_path=labels))
    labels_df['time'] = labels_df['label_path'].apply(get_time_str)

    # fetch corresponding images
    if not augment:
        images_all = glob(os.path.join(image_dir, '*' + image_ext))
        images_all.sort()
        images_df = pandas.DataFrame(dict(image_path=images_all))
        images_df['time'] = images_df['image_path'].apply(get_time_str)
        images_and_labels = pandas.merge(
            labels_df, images_df,
            how='inner', on='time'
        )
    else:
        print("temporal image augmentation not yet implemented. Stopping.")
        return

    # mix order of rows
    images_and_labels = images_and_labels.sample(frac=1, random_state=1).reset_index(drop=True)

    # determine numbers of train, test, and validation
    sample_counts = {
        "train": frac_train*images_and_labels.shape[0]*(1-reduce_train),
        "test": frac_test*images_and_labels.shape[0],
        "validate": frac_validate*images_and_labels.shape[0]
    }
    images_and_labels['role'] = ''
    counter = 0
    for role in ["train", "test", "validate"]:
        images_and_labels.loc[counter:(counter + sample_counts[role]), 'role'] = role
        counter = counter + sample_counts[role]

    # Write to csv file
    csv_file = os.path.join(output_dir, name + '.csv')
    images_and_labels.to_csv(csv_file)

    # Copy images to directories
    # if not os.path.exists(os.path.join(output_dir, name)):
    #     os.makedirs(os.path.join(output_dir, name))
    # for role in ["train", "test", "validate"]:
    #     subdir = os.path.join(output_dir, name, role)
    #     if not os.path.exists(subdir):
    #         os.makedirs(subdir)
    #     for file in images_and_labels.loc[images_and_labels['role']==role, :]:
    #         shutil.copy(file['image_path'], subdir)


def get_time_str(filepath):
    return '_'.join(os.path.splitext(os.path.basename(filepath))[0].split('_')[-3:-1])