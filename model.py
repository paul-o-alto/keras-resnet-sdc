import os
import csv
import time

from scipy import ndimage
import numpy as np
import tensorflow as tf
flags = tf.app.flags
flags.FLAGS.CUDA_VISIBLE_DEVICES = ''
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


DRIVING_TYPES = ['mixed']#, 'corrective', 'inline'] # Choose what type of data to sample
COURSES = ['flat'] #, 'inclines'] # Randomly sample from this
# These two lists represent a tree: top level choose flat or inclines
# Second level choose mixed, corrective or inline

BASE_PATH = '/home/paul/workspace/keras-resnet-sdc/recorded_data'
MINI_BATCH_SIZE = 64

# needed, because mse and mae just produce a network that predicts the mean angle
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=0)

def base_model():
    overall_activation = 'relu' 

    model = Sequential()
    # From Nvidia paper
    #model.add(BatchNormalization(input_shape=(160, 320, 3),
    #                             epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    model.add(Convolution2D(24, 5, 5, input_shape=(160, 320, 3), 
                            border_mode='valid', dim_ordering='tf'))
    model.add(MaxPooling2D(strides=(4,4)))
    model.add(Activation(overall_activation))
    model.add(Convolution2D(36, 5, 5))
    model.add(Activation(overall_activation))
    model.add(Convolution2D(48, 5, 5))
    model.add(Activation(overall_activation))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation(overall_activation))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation(overall_activation))
    model.add(MaxPooling2D(strides=(4,4)))
    model.add(Flatten())
    
    model.add(Dense(1164))
    model.add(Activation(overall_activation))
    model.add(Dropout(0.1))
    model.add(Dense(100))
    model.add(Activation(overall_activation))
    model.add(Dropout(0.1))
    model.add(Dense(50))
    model.add(Activation(overall_activation))
    model.add(Dense(10))

    model.add(Dense(1))
    model.add(Activation(overall_activation)) 

    return model



def load_batch(course_data, course_name, batches_so_far, batch_size, test=False):

    driving_type = np.random.choice(DRIVING_TYPES)
    driving_type_data = course_data[driving_type]

    exhausted = False
    if not test: 
        #coin = 0 #np.random.randint(0,2)
        # Choose which side of the data to grab (0=beginning, 1=ending)
        # Older vs. newer
        if True: #if coin == 0:
            s_index = batches_so_far*batch_size
            if s_index + batch_size*2 > len(driving_type_data):
                s_index = s_index % len(driving_type_data)
                exhausted = True

            sub_list = driving_type_data[s_index:s_index+batch_size]
        #else:
        #    sub_list = driving_type_data[-1:-1-batch_size*2]
        data, labels = build_batch(batch_size, sub_list, course_name, driving_type)
    else:
        batch_size = int(0.3*batch_size)
        sub_list = driving_type_data[0:batch_size] #len(course_data)-1-batch_size:]
        data, labels = build_batch(batch_size, sub_list, course_name, driving_type)

    print("Fetching batch of size %s for the %s course with driving type %s" % (batch_size, 
                                                                      course_name, driving_type))

    return exhausted, data, labels

def build_batch(batch_size, sub_list, course_name, driving_type):

    labels, data = [], []
    current_size = 0
    # Randomly sample from that subset
    stop = False
    while current_size < batch_size:
        row = sub_list[current_size]
        if "," not in row: 
            print ("Invalid csv row, no comma(s)")
            break # Only thing that works
        values = row.split(",")
        if len(values) != 7: 
            print ("Invalid csv line, values != 7")
            continue
        filename_full, _, _, label, _, _, _ = values
        filename_partial = filename_full.split("/")[-1] 
        # above is a hack, because of how Udacity simulator works

        # Randomly choose between mixed, corrective, or inline driving sets
        tmp_img = ndimage.imread(
                        os.path.join(BASE_PATH, course_name, driving_type, 
                                     "IMG", filename_partial))
        data.append(tmp_img)
        labels.append([label])    
        current_size += 1

    data   = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)

    return data, labels            


def load_data(csv_lists, batches_so_far=0, batch_size=512, test=False):

    # Randomly choose between the courses
    course_name = np.random.choice(COURSES)
    
    course_data = csv_lists[course_name]
   
    exhausted, X_train, y_train = load_batch(course_data, course_name, batches_so_far, 
                                  batch_size=batch_size)

    y_train = np.reshape(y_train, (len(y_train), 1))
 
    if test:
        _, X_test, y_test = load_batch(course_data, course_name, batches_so_far, 
                                   batch_size=batch_size, test=True)
        y_test = np.reshape(y_test, (len(y_test), 1))
    else:
        X_test = None
        y_test = None


    return exhausted, (X_train, y_train), (X_test, y_test)


def main():

    mini_batch_size = MINI_BATCH_SIZE
    nb_epoch = 20
    data_augment = False

    model = base_model()
    #model = ResNetBuilder.build_resnet_18((3, 320, 160), 1)
    model.compile(#lr=0.01, 
                  loss=sum_squared_error, #'mse', 
                  optimizer='adam', #'RMSprop', #'adam'
                  ) # RMSprop'),
    model.summary()

    seed = 7
    np.random.seed(seed)
    
    # autosave best Model and load any previous weights
    best_model_file = "./model.h5"
    if os.path.isfile(best_model_file):   
        model.load_weights(best_model_file)
    checkpointer = ModelCheckpoint(best_model_file, 
                                   verbose = 1, save_best_only = False)
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
    csv_lists = {}
    for course in COURSES:
        course_dict = {}
        type_dict = {}
        for driving_type in DRIVING_TYPES:
            f = open(os.path.join(BASE_PATH, course, driving_type, 'driving_log.csv'))
            csv_list = f.read().split("\n")
            type_dict[driving_type] = csv_list
            f.close()
        csv_lists[course] = type_dict
   
    for key in csv_lists.keys():
        print(key)
        for driving_type in csv_lists[key].keys():
            print("  %s" % driving_type)
 
    
 
    print("fitting the model on the batches generated by datagen.flow()")
    exhausted = False # Means we have gone through all of the training set
    batches = 0
    exhausted, (X_train, y_train), (X_test, y_test) = load_data(csv_lists, 
                                                     batches_so_far=batches,
                                                     test=True)
    X_test = X_test.astype('float32')
    X_test = (X_test - np.mean(X_test))/np.std(X_test)
    epoch = 1
    print("Starting first epoch")
    while epoch <= 10:
        batches += 1
        print('Starting batch %s' % batches)
        X_train = X_train.astype('float32')
        X_train = (X_train - np.mean(X_train))/np.std(X_train)

        model.fit_generator(datagen.flow(X_train, y_train, 
                                         batch_size=mini_batch_size),
                            samples_per_epoch=len(X_train), nb_epoch=1, # Epochs on subsample 
                            validation_data=(X_test, y_test), 
                            callbacks=[checkpointer])
                            
        exhausted, (X_train, y_train), _ = load_data(csv_lists, 
                                          batches_so_far=batches)
        
        if exhausted:
            epoch += 1
            print("Starting epoch %s" % epoch)
     

if __name__ == '__main__':
    main()
