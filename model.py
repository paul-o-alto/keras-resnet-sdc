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
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(320, 160, 3),
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
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, init = "normal", subsample= (2, 2), name='conv1_1', 
              input_shape=(320, 160, 3)))
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

def load_batch(fpath, batches_so_far, batch_size, max_size=1000):

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

        for row in reader[s_index:]:

            filename_full, _, _, label, _, _, _ = row.split(",")
            filename_partial = filename_full.split("/")[-1] 
            # above is a hack, because of how Udacity simulator works

            tmp_img = ndimage.imread(
                          os.path.join(fpath, "IMG", filename_partial))
            #tmp_img = tmp_img.transpose((2,0,1))
            #print(tmp_img.shape)
            print("Adding %s to batch" % filename_partial)
            data.append(tmp_img)
            labels.append([label])    
            current_size += 1
            if current_size == max_size: 
                break

    data   = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)


    return data, labels

def load_data(batches_so_far=0, batch_size=32, total_size=1024):

    path = '/home/paul/workspace/keras-resnet-sdc/recorded_data'

    fpath = os.path.join(path, 'train')
    X_train, y_train = load_batch(fpath, batches_so_far, 
                                                batch_size, max_size=total_size/2)

    fpath = os.path.join(path, 'validation')
    X_test, y_test = load_batch(fpath, batches_so_far, batch_size, 
                                             max_size=total_size/8)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    print('X_train dimensions: ', X_train.shape)
    X_train = X_train.transpose((0,2,1,3))
    print('X_train_T dimensions: ', X_train.shape)
    X_test  = X_test.transpose((0,2,1,3))
 
    return (X_train, y_train), (X_test, y_test)


def main():

    learning_rate = 0.1
    batch_size = 32
    nb_epoch = 50
    data_augment = False

    model = steering_net() #base_model()
    model.compile(loss='mean_squared_error', optimizer='adam')

    seed = 7
    np.random.seed(seed)
    
    # autosave best Model
    best_model_file = "./model.h5"
    best_model = ModelCheckpoint(best_model_file, verbose = 1, save_best_only = True)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # this will do preprocessing and realtime data augmentation
    
    train_datagen = ImageDataGenerator(
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
    for e in range(nb_epoch):
        print('Epoch', e)
        batches = 0

        # Fetch a random "Super"-batch for each epoch (some multiple of batch_size below)
        #  X_train, X_test: (nb_samples, 3, 320, 160)
        (X_train, y_train), (X_test, y_test) = load_data(batches_so_far= batches,
                                                         batch_size=batch_size, 
                                                         total_size=1024)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        # perpixel mean substracted
        X_train = (X_train - np.mean(X_train))/np.std(X_train)
        X_test = (X_test - np.mean(X_test))/np.std(X_test)        

        model.fit(X_train, y_train) 
        for X_batch, y_batch in train_datagen.flow(X_train, y_train, batch_size=batch_size):
            loss = model.train(X_batch, y_batch)
            batches += 1
            if batches >= len(X_train) / batch_size:
                # generator loops indefinetly, this is needed
                break


        print('loading best model...')
        model.load_weights(best_model_file)
        score = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
        print('Validaton score:', score)
        print('Validation accuracy:', score[1])


if __name__ == '__main__':
    main()
