# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 3
    n_class = 3
    

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label, image_name = self._next_data()
            
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        
        train_data, labels = self._post_process(train_data, labels)
        
        nx = data.shape[1]
        ny = data.shape[0]

        # reshape
        train_data = train_data.reshape(1, ny, nx, self.channels)
        if labels != '':
            labels = labels.reshape(1, ny, nx, self.n_class)
        return train_data, labels, image_name
    
    def _process_labels(self, label):
        # not necessary with watsen mask formatting (the labels are already in the correct format)
        # if self.n_class == 2:
        #     nx = label.shape[1]
        #     ny = label.shape[0]
        #     labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
        #     labels[..., 1] = label
        #     labels[..., 0] = ~label
        #     return labels
        if label == '':
            return label
        else:
            return label[:, :, :]  # numpy reads rgb bands as bgr, so the order of classes is inverted we return all three layers
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        return data, labels
    
    def __call__(self, n):
        train_data, labels, image_name = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        names = list(np.arange(n))
        X = np.zeros((n, nx, ny, self.channels))
        if labels != '':
            Y = np.zeros((n, nx, ny, self.n_class))
        else:
            Y = list(np.arange(n))
    
        X[0] = train_data
        Y[0] = labels
        names[0] = image_name
        for i in range(1, n):
            train_data, labels, image_name = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
            names[i] = image_name
    
        return X, Y, names

    def _next_data(self):
        return [1], [1], 'test'
    
class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.

    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    
    """
    
    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class = 3):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels = channels

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        return self.data[idx], self.label[idx]


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    
    """
    
    n_class = 3
    
    def __init__(self, images_path, labels_path='', label_validity=10, randomize=False, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif'):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.label_validity = label_validity
        self.file_idx = -1
        self.haslabels = (labels_path is not '')

        self.data_files = []

        if not self.haslabels:
            self.data_files = glob.glob(os.path.join(images_path, "*"+self.data_suffix))
            self.label_files = []

        else:
            #find label files
            self.label_files = glob.glob(os.path.join(labels_path, "*"+self.mask_suffix))
            print(os.path.join(labels_path, "*"+self.mask_suffix))
            print("Number label files found: %s" % len(self.label_files))

            #find image files
            self.data_files = self._find_data_files(images_path)

        # randomize the files if needed or sort them by filename
        if randomize:
            random.shuffle(self.data_files)
        else:
            self.data_files = sorted(self.data_files)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_data_files(self, images_path):
        all_files = glob.glob(os.path.join(images_path, '*'+self.data_suffix))
        return [f for f in all_files if self._getmatchinglabel(f)]
    
    def _getmatchinglabel(self, f):
        filetime = pd.to_datetime(os.path.basename(f).split('_')[0], format='%y%m%d %H%M%S')
        if not self.label_files == []:
            for lf in self.label_files:
                labeltime = pd.to_datetime(os.path.basename(lf).split('_')[0], format='%y%m%d %H%M%S')
                # see if the timedelta is valid
                if abs(labeltime-filetime) <= pd.to_timedelta(self.label_validity/2, 's'):
                    return lf
        else:
            return ''

    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
        
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = self._getmatchinglabel(image_name)
        
        img = self._load_file(image_name, np.float32)
        if label_name != '':
            label = self._load_file(label_name, np.bool)
        else:
            label = ''
    
        return img, label, image_name
