#import all libraries
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D, DepthwiseConv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, SeparableConv2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.optimizers import SGD

import pandas as pd
import cv2
from matplotlib import pyplot as plt
import shutil
import os
K.set_image_dim_ordering('tf')
seed = 7
np.random.seed(seed)
%matplotlib inline

#stuffs related to uploading files from google drive to colab

from google.colab import drive
drive.mount('/content/drive')
!pip install pyunpack
!pip install patool
from pyunpack import Archive
Archive('../content/drive/My Drive/data.tar.gz').extractall('../content/')
!tar -xvf data.tar.gz

df = pd.read_csv('../content/leafsnap-dataset-images.txt', sep='\t')

train_dir = '../content/data/train'
test_dir = '../content/data/test'
validation_dir = '../content/data/validation'


train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
#         horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')
        
validation_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to 
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')
test_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest')

test_generator = test_datagen.flow_from_directory(
        test_dir,  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')
          

# define the model as discussed in https://arxiv.org/pdf/1704.04861.pdf
def mobile_net(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(32, (3, 3), strides = (2, 2), padding = 'same', name = 'mukul_0')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(32,  (3, 3), strides = (1, 1), padding = 'same', depth_multiplier = 1, name = 'jayant_0')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(64, (1, 1), strides = (1, 1), name = 'mukul_1')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(64,  (3, 3), strides = (2, 2), padding = 'same', depth_multiplier = 1, name = 'jayant_1')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(128, (1, 1), strides = (1, 1), name = 'mukul_2')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(128,  (3, 3), strides = (1, 1), padding = 'same', depth_multiplier = 1, name = 'jayant_2')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(128, (1, 1), strides = (1, 1), name = 'mukul_3')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(128,  (3, 3), strides = (2, 2), padding = 'same', depth_multiplier = 1, name = 'jayant_3')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (1, 1), strides = (1, 1), name = 'mukul_4')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(256,  (3, 3), strides = (1, 1), padding = 'same', depth_multiplier = 1, name = 'jayant_4')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (1, 1), strides = (1, 1), name = 'mukul_5')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(256,  (3, 3), strides = (2, 2), padding = 'same', depth_multiplier = 1, name = 'jayant_5')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (1, 1), strides = (1, 1), name = 'mukul_6')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(512,  (3, 3), strides = (1, 1), padding = 'same', depth_multiplier = 1, name = 'jayant_6')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (1, 1), strides = (1, 1), name = 'mukul_7')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(512,  (3, 3), strides = (1, 1), padding = 'same', depth_multiplier = 1, name = 'jayant_7')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (1, 1), strides = (1, 1), name = 'mukul_8')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(512,  (3, 3), strides = (1, 1), padding = 'same', depth_multiplier = 1, name = 'jayant_8')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (1, 1), strides = (1, 1), name = 'mukul_9')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(512,  (3, 3), strides = (1, 1), padding = 'same', depth_multiplier = 1, name = 'jayant_9')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (1, 1), strides = (1, 1), name = 'mukul_10')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(512,  (3, 3), strides = (1, 1), padding = 'same', depth_multiplier = 1, name = 'jayant_10')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (1, 1), strides = (1, 1), name = 'mukul_11')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = SeparableConv2D(512,  (3, 3), strides = (2, 2), padding = 'same', depth_multiplier = 1, name = 'jayant_11')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(1024, (1, 1), strides = (1, 1), name = 'mukul_12')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = AveragePooling2D( (7, 7), strides = (1, 1), name='avg_pool')(X)
    
    X = Flatten()(X)
    
    X = Dense(185, activation = 'softmax', name = 'mahesh')(X)

    model = Model(inputs = X_input, outputs = X, name='mobile_net')
    
    return model
    
model = mobile_net( [224, 224, 1] )
model.summary()
sgd = SGD(lr= 0.01, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit_generator(train_generator,
                       steps_per_epoch=24694//32,
                       validation_data=validation_generator, 
                       epochs=50, 
                       verbose=1,
                       workers=1,
                       use_multiprocessing=False,
                       validation_steps=3090//32 )


evaltest =  model.evaluate_generator(test_generator, 1)
for name, val in zip(model.metrics_names, evaltest):
    print(name, val)
    
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='../content/drive/My Drive/', histogram_freq=0, write_graph=True)
tensorboard.set_model(model)
model.save("leaf.mobilenet.h5")
