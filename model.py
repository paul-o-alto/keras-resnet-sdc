import os
import csv
import time
import cv2

import scipy as sp
import numpy as np
import tensorflow as tf
flags = tf.app.flags
flags.FLAGS.CUDA_VISIBLE_DEVICES = ''
from augmentation import generate_train_from_PD_batch
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from keras import backend as K
from keras.layers.core import Lambda
from keras.layers.advanced_activations import (
    LeakyReLU,
    PReLU,
    ELU
)

from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16, VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.optimizers import SGD, Adam, RMSprop


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
    Dropout,
    SpatialDropout2D,
    Reshape,
    ELU,
    Conv2D,
    GlobalAveragePooling2D
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D
)
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l2

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

from customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids

K.set_image_dim_ordering("tf")

DRIVING_TYPES = ['']
#                 'corrective', 
#                 'inline'
#                ] # Choose what type of data to sample
COURSES = ['']
#           'flat',
#           'inclines'
#          ] # Randomly sample from this
# These two lists represent a tree: top level choose flat or inclines
# Second level choose mixed, corrective or inline
BASE_PATH = '/home/paul/workspace/keras-resnet-sdc/data' #recorded_data'
MINI_BATCH_SIZE = 256
RESIZE_FACTOR = 0.5
EPOCHS = 5

#INPUT_SHAPE = (160*RESIZE_FACTOR, 320*RESIZE_FACTOR, 3)
INPUT_SHAPE = (64,64,3)
WEIGHTS = 'imagenet'
TRAINABLE = False # Determines if we train existing architectures end-to-end
NORMALIZE = False #True # Whether or not to normalize data before it enters the NN
DROPOUT   = True


# needed, because mse and mae just produce a network that predicts the mean angle
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=0)

def tanh_scaled(x):
    return 2*K.tanh(x)

# From Nvidia paper
def nvidia_model():
    overall_activation = 'elu' #'linear' # DO NOT CHANGE! NEEDED IN ORDER TO AVOID SATURATION!
    

    model = Sequential()
    if NORMALIZE:
        model.add(Lambda(lambda x: x/127.5 - 1,
                  input_shape=INPUT_SHAPE))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), 
                            input_shape=INPUT_SHAPE, 
                            border_mode='valid', dim_ordering='tf'))
    model.add(Activation(overall_activation))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Activation(overall_activation))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Activation(overall_activation))
    model.add(Convolution2D(64, 3, 3, border_mode='valid')) 
    model.add(Activation(overall_activation))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation(overall_activation))
    model.add(Flatten())
    model.add(Dense(1164, init="normal"))
    if DROPOUT: model.add(Dropout(0.1)) 
    model.add(Activation(overall_activation))
    model.add(Dense(100, init="normal"))
    if DROPOUT: model.add(Dropout(0.1)) 
    model.add(Activation(overall_activation))
    model.add(Dense(50, init="normal"))
    if DROPOUT: model.add(Dropout(0.1)) 
    model.add(Activation(overall_activation))
    model.add(Dense(10, init="normal"))
    if DROPOUT: model.add(Dropout(0.1)) 
    model.add(Activation(overall_activation))
    model.add(Dense(1))
    model.add(Activation('linear'))
      

    return model


# Highest promise, reasonably small and pretrained...
def vgg16_model():

    input_image = Input(shape=INPUT_SHAPE)
    n_input = Lambda(lambda input_image: input_image/127.5 - 1,
                     input_shape=INPUT_SHAPE)
    base_model = VGG16(weights=WEIGHTS,
                       input_tensor=input_image, 
                       include_top=False)

    if not TRAINABLE:		
        for layer in base_model.layers[:-6]:
            layer.trainable = False
   
    x = base_model.output
    x = Flatten()(x) #GlobalAveragePooling2D()(x)
    x = Dense(4096, activation="relu")(x) #, W_regularizer=l2(0.01))(x)
    x = Dense(4096, activation="relu")(x) #, W_regularizer=l2(0.01))(x)
    x = Dense(1, activation="linear")(x)
    model = Model(input=input_image, output=x)
    return model

def vgg19_model():

    input_image = Input(shape=INPUT_SHAPE)
    base_model = VGG19(weights=WEIGHTS, 
                       input_tensor=input_image, include_top=False)
    if not TRAINABLE:		
        for layer in base_model.layers[:-6]:
            layer.trainable = False
    
    x = base_model.output
    x = Flatten()(x) #GlobalAveragePooling2D()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    return Model(input=input_image, output=x)

def resnet_model():

    input_image = Input(shape=INPUT_SHAPE)
    base_model = ResNet50(input_tensor=input_image,
                          weights=WEIGHTS, 
                          include_top=False)
    if not TRAINABLE:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(100,  activation='relu', name='fc1000')(x)
    pred = Dense(1, activation='relu')(x)
    model = Model(input=base_model.input, output=pred)
    return model


def xception_model():

    base_model = Xception(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1164, init="normal", activation='elu')(x)
    x = Dense(100, init="normal", activation='elu')(x)
    x = Dense(50, init="normal", activation='elu')(x)
    x = Dense(10, init="normal", activation='elu')(x)
    pred = Dense(1)(x) 
    model = Model(input=base_model.input, output=pred)
    return model

def inception_model():

    base_model = InceptionV3(weights=WEIGHTS, include_top=False)
    if not TRAINABLE:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(200,  activation='relu')(x)
    pred = Dense(1)(x)
    model = Model(input=base_model.input, output=pred)
    return model

def comma_ai_model():
    model = Sequential()
    overall_activation = 'elu'

    if NORMALIZE: 
        model.add(Lambda(lambda x: x/127.5 - 1,
                  input_shape=INPUT_SHAPE))
    model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same",
                     input_shape=INPUT_SHAPE))
    model.add(Activation(overall_activation))
    model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation(overall_activation))
    model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same"))

    model.add(Flatten())

    if DROPOUT: model.add(Dropout(0.2))
    model.add(Activation(overall_activation))
    model.add(Dense(512))
    if DROPOUT: model.add(Dropout(0.5))
    model.add(Activation(overall_activation))
    model.add(Dense(1))
    
    return model

def alexnet_model(weights_path=None, regression=True):

    if regression:
        input_ = Input(shape=INPUT_SHAPE)

    model = Sequential()
    model.add(Convolution2D(64, 11, 11, subsample=(4, 4),
                            input_shape=INPUT_SHAPE, border_mode='valid', dim_ordering='tf'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(128, 7, 7, subsample=(3,3), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(192, 3, 3, subsample=(3,3), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(256, 3, 3, subsample=(3,3), border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(8192, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4096, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4096, init='normal'))
    #model.add(BatchNormalization(1000))
    model.add(Dense(1)) #Activation('softmax'))

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
            #print(sub_list)
        #else:
        #    sub_list = driving_type_data[-1:-1-batch_size*2]
        data, labels = build_batch(batch_size, sub_list, course_name, driving_type)
    else:
        batch_size = int(0.3*batch_size)
        sub_list = driving_type_data[-1-batch_size:-1] #len(course_data)-1-batch_size:]
        data, labels = build_batch(batch_size, sub_list, course_name, driving_type)

    print("Fetching batch of size %s for the %s course with driving type %s" % (batch_size, 
                                                                      course_name, driving_type))

    return exhausted, data, labels

def build_batch(batch_size, sub_list, course_name, driving_type):

    labels, data = [], []
    current_size = 1
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
        #print(values)
        tmp_path = os.path.join(BASE_PATH, course_name, driving_type,
                                     "IMG", filename_partial)
        #print("Using path %s", tmp_path)
        tmp_img = cv2.imread(tmp_path)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2YUV)
        if RESIZE_FACTOR < 1:
            tmp_img = cv2.resize(tmp_img, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

        data.append(tmp_img)
        labels.append([label])    
        current_size += 1

    data   = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)

    return data, labels            


def load_data(csv_lists, batches_so_far=0, batch_size=256, test=False):

    # Randomly choose between the courses
    course_name = np.random.choice(COURSES)
    
    course_data = csv_lists[course_name]
   
    exhausted, X_train, y_train = load_batch(course_data, course_name, batches_so_far, 
                                  batch_size=batch_size)

    y_train = np.reshape(y_train, (len(y_train), 1))
 
    if test:
        _, X_test, y_test = load_batch(course_data, course_name, batches_so_far, 
                                   batch_size=batch_size, test=test)
        y_test = np.reshape(y_test, (len(y_test), 1))
    else:
        X_test = None
        y_test = None


    return exhausted, (X_train, y_train), (X_test, y_test)


def main():

    mini_batch_size = MINI_BATCH_SIZE 
    model = 'nvda' 

    if model == 'nvda':
        model = nvidia_model()
    elif model == 'alex':
        model = alexnet_model()
    elif model == 'comma':
        model = comma_ai_model()
    elif model == 'resnet':
        model = resnet_model() # WORST OF THEM ALL!
    elif model == 'inception':
        model = inception_model()
    elif model == 'xception':
        model = xception_model()
    elif model == 'vgg16':
        model = vgg16_model()
    elif model == 'vgg19':
        model = vgg19_model()  

    model.compile(loss='mse', #sum_squared_error,  
                  optimizer='adam') 
    model.summary()
    seed = 7
    np.random.seed(seed)
    
    # autosave best Model and load any previous weights
    model_file = "./model.h5"
    checkpointer = ModelCheckpoint(model_file,
                                   verbose = 1, save_best_only = False) #True)
    if os.path.isfile(model_file):   
        model.load_weights(model_file)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # this will do preprocessing and realtime data augmentation 
    datagen = ImageDataGenerator(
        rescale=1,
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
        horizontal_flip = False,  # randomly flip images
        vertical_flip = False)  # randomly flip images

    print("Start training...")
    csv_lists = {}
    #data_files_s = 
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
    epoch  = 0
    exhaust_batch_amount = 0
    val_size = 1
    print("Starting first epoch")
    print(csv_list[0])
    csv_list = csv_list[1:] # Remove keys   
    print(csv_list[0])

    try: 
        while epoch < EPOCHS: 
            batches += 1
            #print('Starting batch %s' % batches)
            
            train_r_generator = generate_train_from_PD_batch(csv_list, mini_batch_size)
            nb_vals = np.round(len(csv_list)/val_size) - 1
            model.fit_generator(#datagen.flow(X_train, y_train, 
                                train_r_generator,            
                                #batch_size=mini_batch_size),
                                samples_per_epoch=20000, #len(X_train), 
                                nb_epoch=1, # on subsample
                                verbose=1,
                                #validation_data=(X_test, y_test), 
                                callbacks=[checkpointer]
                                )
                            
            exhausted, (X_train, y_train), _ = load_data(csv_lists, batch_size=mini_batch_size, 
                                          batches_so_far=batches)
        
            #if exhausted: 
            #    batches = -1
            epoch += 1
            print("End epoch %s on training set" % epoch)
    except KeyboardInterrupt:
        print("Saving most recent weights before halting...")
        model.save_weights(model_file)      

if __name__ == '__main__':
    main()
