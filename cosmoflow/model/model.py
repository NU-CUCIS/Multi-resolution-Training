'''
Copyright (C) 2020, Northwestern University and Lawrence Berkeley National Laboratory
See COPYRIGHT notice in top-level directory.
'''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

class model ():
    def __init__ (self, num_filters = 16, kernel_size = 3, pool_size = 2):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        print ("Initializing cosmoflow model...")

    def build_model(self):
        x_in = Input(shape = (64, 64, 64, 4), name = "input")
        # The first batch normalization should be replaced with preprocessing later.
        h = BatchNormalization(name = 'bn1')(x_in)

        h = Conv3D(16, self.kernel_size, padding = 'same', activation = 'linear', name = 'conv1')(h)
        h = BatchNormalization(name = 'bn2')(h)
        h = ReLU(name = 'relu1')(h)
        h = AveragePooling3D(pool_size = self.pool_size, name = 'pool1')(h)

        h = Conv3D(32, self.kernel_size, padding = 'same', activation = 'linear', name = 'conv2')(h)
        h = BatchNormalization(name = 'bn3')(h)
        h = ReLU(name = 'relu2')(h)
        h = AveragePooling3D(pool_size = self.pool_size, name = 'pool2')(h)

        h = Conv3D(64, self.kernel_size, padding = 'same', activation = 'linear', name = 'conv3')(h)
        h = BatchNormalization(name = 'bn4')(h)
        h = ReLU(name = 'relu3')(h)
        h = AveragePooling3D(pool_size = self.pool_size, name = 'pool3')(h)

        h = Conv3D(128, self.kernel_size, strides = (2, 2, 2), padding = 'same', activation = 'linear', name = 'conv4')(h)
        h = BatchNormalization(name = 'bn5')(h)
        h = ReLU(name = 'relu4')(h)
        h = AveragePooling3D(pool_size = self.pool_size, name = 'pool4')(h)

        h = Conv3D(256, self.kernel_size, padding = 'same', activation = 'linear', name = 'conv5')(h)
        h = BatchNormalization(name = 'bn6')(h)
        h = ReLU(name = 'relu5')(h)
        h = AveragePooling3D(pool_size = self.pool_size, name = 'pool5')(h)

        h = Conv3D(256, self.kernel_size, padding = 'same', activation = 'linear', name = 'conv6')(h)
        h = BatchNormalization(name = 'bn7')(h)
        h = ReLU(name = 'relu6')(h)

        h = Conv3D(256, self.kernel_size, padding = 'same', activation = 'linear', name = 'conv7')(h)
        h = BatchNormalization(name = 'bn8')(h)
        h = ReLU(name = 'relu7')(h)

        h = Flatten(name = 'pool_to_full')(h)
        h = Dropout(0.05, name='drop1')(h)

        h = Dense(2048, activation = 'linear', name = 'full1')(h)
        h = Dropout(0.05, name='drop2')(h)
        h = ReLU(name = 'relu8')(h)

        h = Dense(256, activation = 'linear', name = 'full2')(h)
        h = Dropout(0.05, name='drop3')(h)
        h = ReLU(name = 'relu9')(h)

        y = Dense(4, activation = 'linear', name = 'full3')(h)
        return Model(inputs = x_in, outputs = y)
