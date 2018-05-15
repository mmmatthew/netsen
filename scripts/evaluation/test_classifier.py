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
                              'M'+os.path.basename(model_dir)+'D'+os.path.splitext(os.path.basename(dataset_csv))[0])

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


def computeIou(dataset, prediction_dir, roles=['test'], channel=None):
    # compute IoU for whole image and for ROI
    data = pandas.read_csv(dataset)
    # filter to roles (train/test)
    data = data.loc[data['role'].isin(roles), :]
    # get prediction paths
    y_pred_paths = pandas.DataFrame(dict(pred_path=glob(prediction_dir + '/*')))
    y_pred_paths['time'] = ['_'.join(os.path.basename(os.path.splitext(p)[0]).split('_')[-3:-1]) for p in y_pred_paths['pred_path']]
    # match two together
    paths = pandas.merge(data, y_pred_paths, how='inner', on='time')
    # extract all image data
    y_pred_batch = np.array([np.array(Image.open(path), dtype=np.float32) for path in paths.loc[:, 'pred_path']])/s.y_scaling
    y_true_batch = np.array([np.array(Image.open(path), dtype=np.float32) for path in paths.loc[:, 'label_path']])
    # Encode the label into one-hot format (e.g. 0,0,1 instead of 2, or 1,0,0 instead of 0)
    y_true_one_hot = (np.arange(s.network['classes']) == y_true_batch[:, :, :, None]).astype(int)
    # prediction is smaller than truth due to scaling, so crop truth to fit. Also return crop margins
    y_true_one_hot_cropped, top_margin, left_margin = crop_truth(y_true_one_hot, y_pred_batch)

    # find out which cam each image comes from
    cam_sources = [os.path.basename(path).split('_')[0] for path in paths.loc[:, 'label_path']]

    # compute pixelwise accuracy image by image and turn into array
    result_overall = np.asarray([pixel_accuracy(
        y_pred_batch[i],
        y_true_one_hot_cropped[i],
        channel,
        cam_sources[i], left_margin, top_margin
    ) for i in range(len(y_true_batch))])
    # find where there are no nans
    result_no_nan = ~np.isnan(result_overall)
    # return mean pixelwise accuracy for pixels that are not nan
    return np.mean(result_overall[result_no_nan[:, 0], 0]), np.mean(result_overall[result_no_nan[:, 1], 1])


def pixel_accuracy(y_pred, y_true, channel=0, camera_name='cam1', left_margin=0, top_margin=0):
    confusion = np.tensordot(y_pred, y_true, axes=([0, 1], [0, 1]))
    roi = s.rois[camera_name]
    confusion_roi = np.tensordot(
        y_pred[roi['top']-1+top_margin: roi['top'] + roi['height'], roi['left']-1+left_margin: roi['left'] + roi['width'], :],
        y_true[roi['top']-1+top_margin: roi['top'] + roi['height'], roi['left']-1+left_margin: roi['left'] + roi['width'], :],
        axes=([0, 1], [0, 1]))
    iou = confusion[channel, channel] / (np.sum(confusion[channel, :]) + np.sum(confusion[channel, :]))
    iou_roi = confusion_roi[channel, channel] / (np.sum(confusion_roi[channel, :]) + np.sum(confusion_roi[channel, :]))
    return iou, iou_roi


def crop_truth(truth, pred):
    # The first and second dimensions need to be cropped
    t_shape = truth.shape
    p_shape = pred.shape
    y_dif = t_shape[1] - p_shape[1]
    x_dif = t_shape[2] - p_shape[2]
    return truth[:, int(y_dif/2):int((y_dif/2 + p_shape[1])), int(x_dif/2):int(x_dif/2 + p_shape[2]), :], int(y_dif/2), int(x_dif/2)
