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
    data = pandas.read_csv(dataset)
    # filter to roles
    data = data.loc[data['role'].isin(roles), :]
    # get prediction paths
    y_pred_paths = pandas.DataFrame(dict(pred_path=glob(prediction_dir + '/*')))
    y_pred_paths['time'] = ['_'.join(os.path.basename(os.path.splitext(p)[0]).split('_')[-3:-1]) for p in y_pred_paths['pred_path']]
    # match two together
    paths = pandas.merge(data, y_pred_paths, how='inner', on='time')
    # extract paths
    y_pred_batch = np.array([np.array(Image.open(path), dtype=np.float32) for path in paths.loc[:, 'pred_path']])/s.y_scaling
    y_true_batch = np.array([np.array(Image.open(path), dtype=np.float32) for path in paths.loc[:, 'label_path']])
    # Encode the label into one-hot
    y_true_one_hot = (np.arange(s.network['classes']) == y_true_batch[:, :, :, None]).astype(int)
    y_true_one_hot_cropped = crop_truth(y_true_one_hot, y_pred_batch)
    # prediction is smaller than truth due to scaling, so crop truth to fit
    img_rows = y_pred_batch.shape[1]
    img_cols = y_pred_batch.shape[2]
    return np.mean(np.asarray([pixel_accuracy(y_pred_batch[i], y_true_one_hot_cropped[i], img_rows, img_cols, channel) for i in range(len(y_true_batch))]))



def pixel_accuracy(y_pred, y_true, img_rows, img_cols, channel=None):
    # if channel is not None:
    #     y_pred = y_pred[..., channel] > 0.8
    #     y_true = y_true[..., channel] > 0.8
    #     union = (y_pred or y_true) and (not y_pred or not y_true)
    #     intersection = (y_pred and y_true) and ()
    #     return 1.0*np.sum(intersection)/np.sum(union)
    # else:
    y_pred = np.argmax(np.reshape(y_pred, [s.network['classes'], img_rows, img_cols]), axis=0)
    y_true = np.argmax(np.reshape(y_true, [s.network['classes'], img_rows, img_cols]), axis=0)
    y_pred = y_pred * (y_true > 0)
    return 1.0 * np.sum((y_pred == y_true) * (y_true > 0)) / np.sum(y_true > 0)


def crop_truth(truth, pred):
    # The first and second dimensions need to be cropped
    t_shape = truth.shape
    p_shape = pred.shape
    y_dif = t_shape[1] - p_shape[1]
    x_dif = t_shape[2] - p_shape[2]
    return truth[:, int(y_dif/2):int((y_dif/2 + p_shape[1])), int(x_dif/2):int(x_dif/2 + p_shape[2]), :]
