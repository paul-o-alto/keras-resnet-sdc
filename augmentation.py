"""
The code in here is based on Vivek Yadav's Medium article entitled:
'An augmentation based deep neural network approach to learn human driving behavior'
"""

import cv2
import numpy as np
import os.path
import math

rows = 160
cols = 320
pr_threshold = 0.05

def augment_brightness_camera_images(image):

    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

def trans_image(image, steer, trans_range):

    # Translation
    tr_x = trans_range*np.random.uniform() - trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform() - 40/2
    Trans_M = np.float32([[1,0,tr_x], [0, 1,tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols,rows))

    return image_tr, steer_ang, tr_x

def add_random_shadow(image):

    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) - (bot_x-top_x)*(Y_m-top_y) >= 0)] = 1
    #random_bright = .25 + .7*np.random.uniform()

    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond2 = shadow_mask == 0
        if np.randomint(2) == 1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image

new_size_col, new_size_row = 64, 64

def preprocessImage(image):

    shape = image.shape
    # note: numpy arrays are (row, col)
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, (new_size_col, new_size_row), interpolation=cv2.INTER_AREA)
    #image = image/255. -.5
    return image

def preprocess_image_file_train(line_data):

    #print(line_data)

    i_lrc = np.random.randint(3)
    if i_lrc == 0:
        path_file = line_data[1].strip() #1 = 'left'
        shift_ang = .25
    if i_lrc == 1:
        path_file = line_data[0].strip() #0 = 'center'
        shift_ang = 0
    if i_lrc == 2:
        path_file = line_data[2].strip() #2 = 'right'
        shift_ang = -.25

    y_steer = float(line_data[3]) + shift_ang # 3 = 'steering'
       
    image = cv2.imread('./data/'+path_file, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, y_steer, tr_x = trans_image(image, y_steer, 100)
    image = augment_brightness_camera_images(image)
    image = preprocessImage(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip == 0:
        image = cv2.flip(image, 1)
        y_steer = -y_steer

    return image, y_steer


def generate_train_from_PD_batch(data, batch_size = 32):

    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)

    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data[i_line]
            line_data = line_data.split(", ")
            if len(line_data) != 7: continue

            keep_pr = 0
            #x, y = preprocess_image_file_train(line_data)
            while keep_pr == 0:
                x,y = preprocess_image_file_train(line_data)
                pr_unif = np.random
                if abs(y) < .1:
                    pr_val = np.random.uniform()
                    if pr_val > pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1

            #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            #y = np.array([[y]])
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering
