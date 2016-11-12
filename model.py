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
    model.add(Convolution2D(24, 5, 5, init = normal_init, subsample= (2, 2), name='conv1_1', 
              input_shape=(160, 320, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init = normal_init, subsample= (2, 2), name='conv2_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, init = normal_init, subsample= (2, 2), name='conv3_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init = normal_init, subsample= (1, 1), name='conv4_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init = normal_init, subsample= (1, 1), name='conv4_2'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init = normal_init, name = "dense_0"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(100, init = normal_init,  name = "dense_1"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(50, init = normal_init, name = "dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(10, init = normal_init, name = "dense_3"))
    model.add(Activation('relu'))
    model.add(Dense(1, init = normal_init, name = "dense_4"))
    model.add(Lambda(atan_layer, output_shape = atan_layer_shape, name = "atan_0"))

    return model

def load_batch(fpath, max_size=1000):

    labels = []
    data = []
    current_size = 0
    with open(os.path.join(fpath,'driving_log.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
        for row in reader:
            print(row)
            filename_full, _, _, label, _, _, _ = row
            filename_partial = filename_full.split("/")[-1] 
            # above is a hack, because of how simulator works
            tmp_img = ndimage.imread(
                          os.path.join(fpath, "IMG", filename_partial))
            print(tmp_img.shape)
            #tmp_img = tmp_img.transpose((2,0,1))
            #print(tmp_img.shape)
            data.append(tmp_img)
            labels.append([label])    
            current_size += 1
            if current_size == max_size: break

    data   = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)


    return data, labels

def load_data(total_size=100):

    path = '/home/paul/workspace/keras-resnet-sdc/recorded_data'

    fpath = os.path.join(path, 'train')
    X_train, y_train = load_batch(fpath,max_size=total_size*0.8)

    fpath = os.path.join(path, 'test')
    X_test, y_test = load_batch(fpath, max_size=total_size*0.2)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    print('X_train dimensions: ', X_train.shape)
    #X_train = X_train.transpose((0,3,2,1))
    #print('X_train_T dimensions: ', X_train.shape)
    #X_test  = X_test.transpose((0,3,2,1))
 
    return (X_train, y_train), (X_test, y_test)


def main():
    #import time
    #start = time.time()
    #model = base_model()
    #duration = time.time() - start
    #print("{} s to make model".format(duration))

    #start = time.time()
    #model.output
    #duration = time.time() - start
    #print("{} s to get output".format(duration))

    #start = time.time()
    #model.compile(loss="categorical_crossentropy", optimizer="sgd")
    #duration = time.time() - start
    #print("{} s to get compile".format(duration))

    learning_rate = 0.1
    batch_size = 16
    nb_epoch = 25
    data_augment = False

    #  X_train, X_test: (nb_samples, 3, 320, 160)
    (X_train, y_train), (X_test, y_test) = load_data(total_size=1000)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # perpixel mean substracted
    X_train = (X_train - np.mean(X_train))/np.std(X_train)
    X_test = (X_test - np.mean(X_test))/np.std(X_test)


    model = steering_net() #base_model()
    model.compile(loss='mean_squared_error', optimizer='adam')

    seed = 7
    np.random.seed(seed)
    #estimators = []
    #estimators.append(('standardize', StandardScaler()))
    #estimators.append(('mlp', KerasRegressor(build_fn=model, 
    #                                    nb_epoch=50, batch_size=5, verbose=0)))
    #pipeline = Pipeline(estimators)
    #kfold = KFold(n_splits=10, random_state=seed)
    #results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
    #print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    
    # autosave best Model
    best_model_file = "./model.h5"
    best_model = ModelCheckpoint(best_model_file, verbose = 1, save_best_only = True)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # this will do preprocessing and realtime data augmentation
    
    datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip = False)  # randomly flip images

    #datagen.fit(X_train)

    print("Start training...")
    #for i in range(3):
        #if i != 0:
        #    # devide the learning rate by 10 for two times
        #    lr_old = K.get_value(optimizer.lr)
        #    K.set_value(optimizer.lr, 0.1 * lr_old)
        #    print('Changing learning rate from %f to %f' % (lr_old, K.get_value(optimizer.lr)))

    if data_augment:
        print("fitting the model on the batches generated by datagen.flow()")
        model.fit_generator(datagen.flow(X_train, y_train,
                            batch_size = batch_size),
                                samples_per_epoch = X_train.shape[0],
                                nb_epoch = nb_epoch,
                                validation_data = (X_test, y_test),
                                callbacks = [best_model])
    else:
        model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch,
                      verbose = 1, validation_data = (X_test, y_test), callbacks = [best_model])

    print('loading best model...')
    model.load_weights(best_model_file)
    score = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
    print('Test score:', score)
    #print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
