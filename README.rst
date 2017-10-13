=============================
WATSEN: Water Segmentation with Neural Networks
=============================

A neural network approach is proposed as a solution to automatically transform CCTV video sequences to information about the local flood level evolution during a flood event.
The use of a neural network is expected to be well suited to the low video quality and diversity of appearances that water can have.
The method proposed is evaluated with data from `controlled flood experiments <https://www.earth-syst-sci-data.net/9/657/2017/essd-9-657-2017.html>`_, in which video and sensor data were collected in parallel.

The network architecture is that of **U-Net** as proposed by `Ronneberger et al. <https://arxiv.org/pdf/1505.04597.pdf>`_.
The code is largely based on code been developed and used for `Radio Frequency Interference mitigation using deep convolutional neural networks <http://arxiv.org/abs/1609.09077>`_ .
See `the original GitHub repo <https://github.com/jakeret/tf_unet>`_ for more information.
