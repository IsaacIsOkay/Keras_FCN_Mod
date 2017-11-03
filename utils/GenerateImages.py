import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam, Adadelta
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util

from models import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
import time

target_size = (320, 320)
label_cval=100
loss_shape = None
ignore_label = 100
numImages = 100 #num images generated is numImages * 100
datagen = SegDataGenerator(#zoom_range=[0.5, 2.0],
                                     zoom_maintain_shape=True,
                                     #crop_mode='random',
                                     crop_size=target_size,
                                     # pad_size=(505, 505),
                                     #rotation_range=60,
                                     #shear_range=0,
                                     #horizontal_flip=True,
                                     #channel_shift_range=1.,
                                     fill_mode='constant',
                                     label_cval=label_cval,
				     )

train_file_path = os.path.expanduser('~/.keras/datasets/weedspic/lettuce/train.txt') 
val_file_path   = os.path.expanduser('~/.keras/datasets/weedspic/lettuce/validation.txt')
data_dir        = os.path.expanduser('~/.keras/datasets/weedspic/lettuce/image')
label_dir       = os.path.expanduser('~/.keras/datasets/weedspic/lettuce/label')
data_suffix='.png'
label_suffix='.png'
classes = 2


i = 0
for batch in datagen.flow_from_directory(
            	file_path=train_file_path,
            	data_dir=data_dir, data_suffix=data_suffix,
            	label_dir=label_dir, label_suffix=label_suffix,
            	classes=classes,
            	target_size=target_size, color_mode='rgb',
            	batch_size=100, shuffle=True,
            	loss_shape=loss_shape,
            	ignore_label=ignore_label):
 	i+=1
	print(i)
	if i > numImages:
	  break

