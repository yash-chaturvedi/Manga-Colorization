import os
import cv2 
import numpy as np
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import *
import matplotlib.pyplot as plt
from keras.regularizers import *
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from time import time
from keras.preprocessing.image import ImageDataGenerator
import base64
import io
from PIL import Image

from flask import Flask

from flask import jsonify

from flask import request

import json as json

from flask_cors import CORS, cross_origin



app = Flask(__name__)

cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

def get_model():
    global model
    model = load_model('gen.h5',compile=False)
    print('Model loaded')

def preprocess(img):
    x_shape = 512
    y_shape = 512
    I = cv2.resize(img, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('',J)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    # if not cv2.imwrite('D:/Jupyter/GrayScale/2.jpg', J):
    #  raise Exception("Could not write image")
    J = J.reshape(1,J.shape[0], J.shape[1], 1)
    gray_val = J
    return gray_val

print("Loading model...")
get_model()

@app.route('/predict',methods=['POST'])

@cross_origin()

def predict():
    file = request.files['image'].read() ## byte file
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

    processed_img = preprocess(img)

    p = model.predict(processed_img)
    prediction = np.squeeze(p,axis=0)

    dist1 = cv2.convertScaleAbs(prediction)
    # dist2 = cv2.normalize(prediction, None ,0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # cv2.imshow('dist1',dist1)
    # cv2.imshow('dist2',dist2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    prediction = cv2.cvtColor(dist1, cv2.COLOR_BGR2RGB)
    # prediction = Image.fromarray(prediction.astype("uint8"))
    prediction = Image.fromarray(prediction)
    # prediction.show()

    rawBytes = io.BytesIO()
    prediction.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status':str(img_base64)})

















