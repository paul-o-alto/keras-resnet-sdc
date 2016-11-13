import os
import csv
import time

from scipy import ndimage
import numpy as np
import tensorflow as tf
from resnet import ResNetBuilder

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

COURSES = ['flat', 'inclines'] # Randomly sample from this
BASE_PATH = '/home/paul/workspace/keras-resnet-sdc/recorded_data'

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
    model.add(Dense(1, activation='tanh', name = "dense_4")) 

    return model

def load_batch(csv_lists, batches_so_far, batch_size, test=False):

    batch_size = int(batch_size)
    if batch_size == 0: batch_size = 1
    print("Fetching batch of size %s" % batch_size)
    labels = []
    data = []
    current_size = 0

  
    # Randomly choose between the courses
    course_index = np.random.randint(0, len(csv_lists)) 
    # We need randint so that we can index into COURSES list for path
    course_list = csv_lists[course_index]
    course_name = COURSES[course_index]

    # Choose a subset of that course list
    if test:
        # Grab test data from the end
        s_index = len(course_list)-1-batch_size
        # Randomly sample "batch_size" amount from 10*batch_size amount
        sub_list = course_list[s_index:]
    else:
        s_index = batches_so_far*batch_size
        e_index = s_index + batch_size
        if e_index >= len(course_list): 
            s_index = s_index % len(course_list)
        e_index = s_index + batch_size
        sub_list = course_list[s_index:e_index]

    # Randomly sample from that subset
    while current_size < batch_size:  
        row = sub_list[current_size]
        values = row.split(",")
        if len(values) != 7:
            continue
        filename_full, _, _, label, _, _, _ = values
        filename_partial = filename_full.split("/")[-1] 
        # above is a hack, because of how Udacity simulator works

        tmp_img = ndimage.imread(
                        os.path.join(BASE_PATH, course_name, 
                                     "IMG", filename_partial))
        data.append(tmp_img)
        labels.append([label])    
        current_size += 1
            

    data   = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)


    return data, labels

def load_data(csv_lists, batches_so_far=0, batch_size=2048, test=True):

   
    X_train, y_train = load_batch(csv_lists, batches_so_far, 
                                             batch_size=batch_size)

    y_train = np.reshape(y_train, (len(y_train), 1))
 
    if test:
        X_test, y_test = load_batch(csv_lists, batches_so_far, 
                                   batch_size=int(batch_size*0.3), test=True)
        y_test = np.reshape(y_test, (len(y_test), 1))
    else:
        X_test = None
        y_test = None


    return (X_train, y_train), (X_test, y_test)


def main():

    learning_rate = 0.01
    mini_batch_size = 16
    nb_epoch = 25
    data_augment = False

    model = steering_net() #base_model()
    #model = ResNetBuilder.build_resnet_18((3, 320, 160), 1)
    model.compile(lr=learning_rate, loss='mse', optimizer='RMSprop'),
    model.summary()

    seed = 7
    np.random.seed(seed)
    
    # autosave best Model and load any previous weights
    best_model_file = "./model.h5"
    if os.path.isfile(best_model_file):   
        model.load_weights(best_model_file)
    checkpointer = ModelCheckpoint(best_model_file, 
                                   verbose = 1, save_best_only = True)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    if os.path.isfile(best_model_file): 
        model.load_weights(best_model_file)

    # this will do preprocessing and realtime data augmentation 
    datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)
        horizontal_flip = False,  # randomly flip images
        vertical_flip = False)  # randomly flip images


    

    print("Start training...")
    csv_lists = []
    for course in COURSES:
        f = open(os.path.join(BASE_PATH, course, 'driving_log.csv'))
        csv_list = f.read().split("\n")
        csv_lists.append(csv_list)
        f.close()
    
    print("fitting the model on the batches generated by datagen.flow()")
    exhausted = False # Means we have gone through all of the training set
    batches = 0
    (X_train, y_train), (X_test, y_test) = load_data(csv_lists, 
                                                     batches_so_far=batches)
    X_test = X_test.astype('float32')
    X_test = (X_test - np.mean(X_test))/np.std(X_test)
    count = 0
    while True:
        batches += 1
        X_train = X_train.astype('float32')
        X_train = (X_train - np.mean(X_train))/np.std(X_train)

        model.fit_generator(datagen.flow(X_train, y_train, 
                                         batch_size=mini_batch_size),
                            samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
                            validation_data=(X_test, y_test), 
                            callbacks=[checkpointer])
                            
        (X_train, y_train), _ = load_data(batches_so_far=batches, test=False)

if __name__ == '__main__':
    main()
