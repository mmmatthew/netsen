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
        labels = labels.reshape(1, ny, nx, self.n_class)
        return train_data, labels, image_name
    
    def _process_labels(self, label):
        # encode label as one-hot
        one_hot = (np.arange(self.n_class) == label[:, :, None]).astype(int)

        return np.squeeze(one_hot)
    
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
        Y = np.zeros((n, nx, ny, self.n_class))
    
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
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
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

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param n_class: (optional) number of classes, default=2
    
    """

    def __init__(self, dataset, roles=['train'], shuffle_data=False, a_min=None, a_max=None, n_class=3, n_channels=3):
        super(ImageDataProvider, self).__init__(a_min, a_max)

        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.n_channels = n_channels

        # get data paths
        self.data = pd.read_csv(dataset)

        # extract data for current roles
        self.data = self.data.loc[self.data['role'].isin(roles), :]

        # randomize the files if needed or sort them by filename
        if self.shuffle_data:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        else:
            self.data.sort_values(by='time')

        assert len(self.data.index) > 0, "No training files"
        print("Number of files used: %s" % len(self.data.index))

        img = self._load_file(self.data.loc[0, 'image_path'])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data.index):
            self.file_idx = 0
            # reshuffle data
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        
    def _next_data(self):
        self._cylce_file()
        image_path, label_path = self.data.loc[self.file_idx, ['image_path', 'label_path']]

        image = self._load_file(image_path, np.float32)
        label = self._load_file(label_path, np.int8)
    
        return image, label, os.path.basename(image_path)


class ImageDataNoLabelProvider(BaseDataProvider):
    """
    Generic data provider for images without labels

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param n_class: (optional) number of classes, default=2

    """

    def __init__(self, directory, pattern='*.jpg', shuffle_data=False, a_min=None, a_max=None, n_channels=3):
        super(ImageDataProvider, self).__init__(a_min, a_max)

        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_channels = n_channels

        # get data paths
        self.data = glob.glob(os.path.join(directory, pattern))

        # randomize the files if needed or sort them by filename
        if self.shuffle_data:
            self.data = random.shuffle(self.data)
        else:
            self.data.sort()

        assert len(self.data) > 0, "No training files"
        print("Number of files used: %s" % len(self.data))

        img = self._load_file(self.data[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def __call__(self, n):
        train_data, image_name = self._load_data()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        names = list(np.arange(n))
        X = np.zeros((n, nx, ny, self.channels))

        X[0] = train_data
        names[0] = image_name
        for i in range(1, n):
            train_data, image_name = self._load_data()
            X[i] = train_data
            names[i] = image_name

        return X, names

    def _load_data(self):
        data, image_name = self._next_data()

        train_data = self._process_data(data)

        nx = data.shape[1]
        ny = data.shape[0]

        # reshape
        train_data = train_data.reshape(1, ny, nx, self.channels)
        return train_data, image_name

    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data):
            self.file_idx = 0
            # reshuffle data
            self.data = random.shuffle(self.data)

    def _next_data(self):
        self._cylce_file()
        image_path = self.data[self.file_idx]

        image = self._load_file(image_path, np.float32)

        return image, os.path.basename(image_path)
