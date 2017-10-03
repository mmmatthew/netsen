


import os
from glob import glob
import shutil
masks = glob('2_cl/*_mask.tiff')
templates = [os.path.basename(m).split('_')[0] for m in masks]

for t in templates:
  file = glob('F:/MOY_PhD_DATA_LOCAL/2017_watsen/data/frames/'+t+'*.tif')
  if file !=[]:
  	shutil.copy(file[0], '3_cl')
