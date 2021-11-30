#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Definitions, paths and functions for the segmenter and annotator
"""


# %matplotlib inline
from IPython.display import clear_output

import sys
import os
import numpy as np
import pandas as pd
import glob
import h5py
import time
from distutils.dir_util import copy_tree
from shutil import copyfile, copy, copy2


from scipy.ndimage.filters import median_filter
from ImageStackPy import Img_Viewer as VIEW

import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tiff


from ct_segnet import viewer


from ImageStackPy import ImageProcessing as IP

from ipywidgets import interact
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout
from IPython.display import display, update_display
from ipyfilechooser import FileChooser
from skimage.io import imread

import cv2, os, h5py, collections, sys, math
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import clear_output
from ipywidgets import interactive
from scipy import ndimage
from scipy.ndimage import label, generate_binary_structure,measurements,morphology 
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import data
from skimage.filters import threshold_otsu


import sys
if '../' not in sys.path:
    sys.path.append('../')

from tools.misc.Utils import CenterSampling, ClusterAnalysis
from tools.misc.patches2d import Patches2D




# Set the directory where the FILEChooser widget will open up to read .h5 files contating xrf images


annot_dir='annotated_XRF'
base__dir_path=os.path.join(os.path.join(os.path.dirname(os.getcwd()),annot_dir), 'raw_cells')
h5_dir = base__dir_path
default_path = h5_dir 






def text_width(wd):
    return Layout(width = "%ipx"%wd)

