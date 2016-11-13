import os
import csv

from scipy import ndimage
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint

from keras.models import (
    Model,
    Sequential
)
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

def normal_init(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return K.variable(initial)

def atan_layer(x):
    return tf.mul(tf.atan(x), 2)

def atan_layer_shape(input_shape):
    return input_shape

def base_model(): 
    #input = Input(shape=(3, 320, 160))

    model = Sequential()
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(160, 320, 3),
                            dim_ordering='tf'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3))) # 2, 2
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(64))#256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='normal'))
    

    model = Model(input=input, output=dense)
    return model


def steering_net():
    p = 0.1 # Prob of dropout

    model = Sequential()
    model.add(Convolution2D(24, 5, 5, init = "normal", subsample= (2, 2), name='conv1_1', 
              input_shape=(160, 320, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init = "normal", subsample= (2, 2), name='conv2_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, init = "normal", subsample= (2, 2), name='conv3_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init = "normal", subsample= (1, 1), name='conv4_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init = "normal", subsample= (1, 1), name='conv4_2'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init = "normal", name = "dense_0"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(100, init = "normal",  name = "dense_1"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(50, init = "normal", name = "dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(10, init = "normal", name = "dense_3"))
    model.add(Activation('relu'))
    #model.add(Dense(1, init = "normal", name = "dense_4"))
    #model.add(Lambda(atan_layer, output_shape = atan_layer_shape, name = "atan_0"))
    model.add(Dense(1, activation='sigmoid', name = "dense_4")) 

    return model

def load_batch(fpath, batches_so_far, batch_size):

    exhausted = False
    batch_size = int(batch_size)
    if batch_size == 0: batch_size = 1
    print("Fetching batch of size %s" % batch_size)
    labels = []
    data = []
    current_size = 0
    with open(os.path.join(fpath,'driving_log.csv'), newline='') as csvfile:
        
        reader = csvfile.read().split("\n")
        s_index = batches_so_far*batch_size
        e_index = s_index + batch_size
        if e_index > len(reader): 
            s_index = s_index % len(reader)
            exhausted = True

        for row in reader[s_index:]:

            values = row.split(",")
            if len(values) != 7:
                continue
            filename_full, _, _, label, _, _, _ = values #row.split(",")
            filename_partial = filename_full.split("/")[-1] 
            # above is a hack, because of how Udacity simulator works

            tmp_img = ndimage.imread(
                          os.path.join(fpath, "IMG", filename_partial))
            data.append(tmp_img)
            labels.append([label])    
            current_size += 1
            if current_size == batch_size: 
                break

    data   = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)


    return exhausted, data, labels

def load_data(batches_so_far=0, batch_size=1024, test=True):

    path = '/home/paul/workspace/keras-resnet-sdc/recorded_data'
    exhausted = False

    fpath = os.path.join(path, 'train')
    exhausted, X_train, y_train = load_batch(fpath, batches_so_far, 
                                                batch_size=batch_size-200)

    y_train = np.reshape(y_train, (len(y_train), 1))
 
    if test:
        fpath = os.path.join(path, 'validation')
        _, X_test, y_test = load_batch(fpath, batches_so_far, 
                                   batch_size=824)
        y_test = np.reshape(y_test, (len(y_test), 1))
        # We don't care about exhausting the test set, so we discard the value
    else:
        X_test = None
        y_test = None


    return exhausted, (X_train, y_train), (X_test, y_test)


def main():

    learning_rate = 0.1
    mini_batch_size = 16
    nb_epoch = 10
    data_augment = False

    model = steering_net() #base_model()
    model.compile(loss='mean_absolute_error', optimizer='adam'),

    seed = 7
    np.random.seed(seed)
    
    # autosave best Model
    best_model_file = "./model.h5"
    checkpointer = ModelCheckpoint(best_model_file, verbose = 1, save_best_only = True)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # this will do preprocessing and realtime data augmentation
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)
        #width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        #height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  # randomly flip images
        vertical_flip = False)  # randomly flip images


    

    print("Start training...")

    
    print("fitting the model on the batches generated by datagen.flow()")
    exhausted = False # Means we have gone through all of the training set
    batches = 0
    exhausted, (X_train, y_train), (X_test, y_test) = load_data(batches_so_far=batches)
    X_test = X_test.astype('float32')
    X_test = (X_test - np.mean(X_test))/np.std(X_test)
    while not exhausted:
        batches += 1
        X_train = X_train.astype('float32')
        X_train = (X_train - np.mean(X_train))/np.std(X_train)

        model.fit_generator(datagen.flow(X_train, y_train, batch_size=mini_batch_size),
                            samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
                            validation_data=(X_test, y_test), callbacks=[checkpointer])
                            
        exhausted, (X_train, y_train), _ = load_data(batches_so_far=batches, test=False)

if __name__ == '__main__':
    main()
