"""
These functions allows us to test the trained classifiers on data intra-event, inter-event, and inter-camera
"""
from tf_unet import unet
import evaluation_settings as s
import os
from glob import glob
import shutil
import numpy as np
from PIL import Image
import pandas


def test(model_dir, dataset_csv, working_dir, force=False):
    # create folder name
    output_dir = os.path.join(working_dir, s.stages[5],
                              'M' + os.path.basename(model_dir) + 'D' + os.path.splitext(os.path.basename(dataset_csv))[
                                  0])

    # only retest if necessary
    if not os.path.exists(output_dir) or force:
        # delete existing
        if force and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(e)

        # make directory
        os.makedirs(output_dir)

        net = unet.Unet(
            channels=s.network['channels'],
            n_class=s.network['classes'],
            layers=s.network['layers'],
            features_root=s.network['features_root'],
            cost_kwargs=dict(class_weights=s.network['class_weights'])
        )
        # Run prediction
        net.predictAll(
            model_path=os.path.join(model_dir, 'model.cpkt'),
            dataset_path=dataset_csv, output_dir=output_dir, roles=['test'])
    else:
        print(os.path.basename(output_dir), ' already exists. Skipping.')


def computeIou_all(dataset_path, prediction_dir, roles=['test'], channel=None):
    # compute IoU for whole image and for ROI
    dataset = pandas.read_csv(dataset_path)
    # filter to roles (train/test)
    dataset = dataset.loc[dataset['role'].isin(roles), :]

    # get paths to files of predicted flooding coverage
    y_pred_paths = pandas.DataFrame(dict(pred_path=glob(prediction_dir + '/*')))
    # extract time from paths
    y_pred_paths['time'] = ['_'.join(os.path.basename(os.path.splitext(p)[0]).split('_')[-3:-1]) for p in
                            y_pred_paths['pred_path']]

    # match the predictions to the image and label paths based on the time
    paths = pandas.merge(dataset, y_pred_paths, how='inner', on='time')

    # for each image in the dataset, read image data and compute IoU
    for index, row in paths.iterrows():
        paths.loc[index, 'iou'], paths.loc[index, 'iou_roi'] = computeIou_single(label_path=row['label_path'], pred_path=row['pred_path'])

    # write data to file
    paths.to_csv(os.path.join(prediction_dir, 'iou.csv'))

    return paths['iou'].mean(), paths['iou_roi'].mean()


def computeIou_single(label_path, pred_path, channel=2):
    # read image data
    label = np.array(Image.open(label_path), dtype=np.float32)
    pred = np.array(Image.open(pred_path), dtype=np.float32) / s.y_scaling

    # Encode label into one-hot format (e.g. 0,0,1 instead of 2, or 1,0,0 instead of 0)
    label_one_hot = (np.arange(s.network['classes']) == label[:, :, None]).astype(float)
    # predictions are probabilistic - make the highest probability one
    pred_one_hot = (np.arange(s.network['classes']) == (pred.argmax(axis=2))[:, :, None]).astype(float)

    # prediction is smaller than truth due to scaling, so crop truth to fit. Also return crop margins
    label_one_hot, top_margin, left_margin = crop_truth(label_one_hot, pred_one_hot)

    # get camera name and ROI information
    roi = s.rois[os.path.basename(pred_path).split('_')[0]]
    left = max(roi['left'], 0)
    top = max(roi['top'], 0)
    right = min(roi['left'] + roi['width'], pred_one_hot.shape[1])
    bottom = min(roi['top'] + roi['height'], pred_one_hot.shape[0])

    # crop out roi
    pred_one_hot_roi = pred_one_hot[top: bottom, left: right, :]
    label_one_hot_roi = label_one_hot[top: bottom, left: right, :]

    # compute confusion matrix
    confusion = np.tensordot(pred_one_hot, label_one_hot, axes=([0, 1], [0, 1]))
    confusion_roi = np.tensordot(pred_one_hot_roi, label_one_hot_roi, axes=([0, 1], [0, 1]))

    #compute iou union
    union = (np.sum(confusion[channel, :]) + np.sum(confusion[:, channel])) - confusion[channel, channel]
    union_roi = (np.sum(confusion_roi[channel, :]) + np.sum(confusion_roi[:, channel])) - confusion_roi[channel, channel]

    # compute iou (return 1 if union is zero)
    if union == 0:
        iou = 1
    else:
        iou = confusion[channel, channel] / union
    if union_roi == 0:
        iou_roi = 1
    else:
        iou_roi = confusion_roi[channel, channel] / union_roi

    return iou, iou_roi

def crop_truth(truth, pred):
    # Crop the label to the size of the prediction
    t_shape = truth.shape
    p_shape = pred.shape
    y_dif = t_shape[0] - p_shape[0]
    x_dif = t_shape[1] - p_shape[1]
    return truth[int(y_dif / 2):int((y_dif / 2 + p_shape[0])), int(x_dif / 2):int(x_dif / 2 + p_shape[1]), :], int(
        y_dif / 2), int(x_dif / 2)
