import argparse
import base64
import json

import numpy as np
import scipy as sp
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
#from model import tanh_scaled

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from model import RESIZE_FACTOR
from augmentation import preprocessImage
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
DESIRED_SPEED = 10

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    #if RESIZE_FACTOR < 1:
    #    image_array = sp.misc.imresize(image_array, RESIZE_FACTOR)
    image_array = preprocessImage(image_array)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes features of the model are just the images. 
    # Feel free to change this.
    print ("Predicting steering angle")
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    print ("Predicted %s" % steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = (DESIRED_SPEED-abs(float(speed)))*0.5
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read()) #, custom_objects={'tanh_scaled': tanh_scaled})

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
