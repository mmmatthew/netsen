import pandas
import os
from glob import glob
import evaluation_settings as s


def create_all(working_dir, augment=False, augment_duration=0, reduce_train=0.0,
               label_pattern='*.png', image_pattern='*.jpg', force=False):
    """
    Given directories of labeled images, divide into sets for training, testing, and validation.
    Options: natural data augmentation, train/test/validation fractions, reduction in training size
    :param working_dir: working directory of project
    :param augment: augment the dataset by increasing the validity period of each label
    :param augment_duration: duration of the validity period
    :param reduce_train: reduce the fraction of training data by this fraction (the images left out are not used at all)
    :param label_pattern: pattern for discovering label files in folder
    :param image_pattern: pattern for discovering image files in folder
    :return: path to csv file
    """

    # directory containing labeled images
    label_dir = os.path.join(working_dir, s.stages[2])
    # directory containing images (not labeled)
    image_dir = os.path.join(working_dir, s.stages[1])
    # where the dataset definition should be stored
    output_dir = os.path.join(working_dir, s.stages[3])

    datasets = []

    for label_subdir in next(os.walk(label_dir))[1]:
        camera, mode = label_subdir.split('_')
        frac_train = s.datasets[mode]['frac_train']
        frac_validate = s.datasets[mode]['frac_validate']
        frac_test = s.datasets[mode]['frac_test']
        for image_subdir in next(os.walk(image_dir))[1]:
            camera2, multitime = image_subdir.split('_', 1)
            if camera != camera2:
                continue
            else:
                name = '_'.join([camera, mode, multitime])
                datasets.append(create(label_dir=os.path.join(label_dir, label_subdir), image_dir=os.path.join(image_dir,image_subdir), output_dir=output_dir, name=name,
                                       augment=augment, augment_duration=augment_duration, frac_train=frac_train,
                                       frac_test=frac_test, frac_validate=frac_validate, reduce_train=reduce_train,
                                       label_pattern=label_pattern, image_pattern=image_pattern))
    return datasets


def create_combinations(c1, c2, working_dir):
    """Create merged datasets for select cameras c1 and c2, e.g. cam1 and cam5"""
    datasets = pandas.DataFrame(dict(dir=os.listdir(os.path.join(working_dir, s.stages[3]))))
    #divide into parts
    datasets['camera'] = [d.split('_', 1)[0] for d in datasets.dir]
    datasets['rest'] = [d.split('_', 1)[1] for d in datasets.dir]

    c1 = datasets.loc[datasets.camera == c1, :]
    c2 = datasets.loc[datasets.camera == c2, :]
    merged = pandas.merge(c1, c2, on='rest', suffixes=('_1', '_2'))

    # go through and create merged datasets
    for index, row in merged.iterrows():
        # create merged name
        merged_name = row.camera_1 + row.camera_2 + '_' + row.rest
        if not os.path.exists(os.path.join(working_dir, s.stages[3], merged_name)):
            pandas.concat([
                pandas.read_csv(os.path.join(working_dir, s.stages[3], row.dir_1)),
                pandas.read_csv(os.path.join(working_dir, s.stages[3], row.dir_2))
            ]).to_csv(os.path.join(working_dir, s.stages[3], merged_name))


def create(label_dir, image_dir, output_dir, name, augment=False, augment_duration=0, frac_train=0.7, frac_test=0.3,
           frac_validate=0.0, reduce_train=0.0, label_pattern='*.png', image_pattern='*.jpg', force=False):
    # fetch list of label images
    labels = glob(os.path.join(label_dir, 'labels', label_pattern))
    if len(labels) == 0:
        return
    labels.sort()
    labels_df = pandas.DataFrame(dict(label_path=labels))
    labels_df['time'] = labels_df['label_path'].apply(get_time_str)

    # fetch corresponding images
    if not augment:
        images_all = glob(os.path.join(image_dir, image_pattern))
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
        "train": frac_train * images_and_labels.shape[0] * (1 - reduce_train),
        "test": frac_test * images_and_labels.shape[0],
        "validate": frac_validate * images_and_labels.shape[0]
    }
    images_and_labels['role'] = ''
    counter = 0
    for role in ["train", "test", "validate"]:
        images_and_labels.loc[counter:(counter + sample_counts[role]), 'role'] = role
        counter = counter + sample_counts[role]

    # Write to csv file
    csv_file = os.path.join(output_dir, name + '.csv')
    if not os.path.exists(csv_file) or force:
        images_and_labels.to_csv(csv_file)

    return csv_file

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
