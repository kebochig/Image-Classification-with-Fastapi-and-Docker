import uvicorn
import numpy as np
import logging
import PIL
import PIL.Image
import pathlib
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import *
from fastapi import FastAPI
from io import BytesIO

app = FastAPI()

# Load model
loaded_model = tf.keras.models.load_model('my_model.h5')

def predict_img_file(path):
    '''
    This Function predicts image from file path

    Args:
    Filepath

    Returns:
    String describing image
    '''
    try:
        img_path = pathlib.Path(path)
        img = image.load_img(img_path, target_size=(160, 160))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        predictions = loaded_model.predict_on_batch(img_batch).flatten()
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)
        labels = ['Defective Box', 'Non Defective Box']
        return {'prediction': 'This is a {}'.format(labels[predictions.numpy()[0]])}
    except Exception as ex:
        logging.info("Error:", ex)
        exit('Error Occured: Check File Path')


def predict_img_url(url):
    '''
    This Function predicts image from url

    Args:
    Filepath

    Returns:
    String describing image
    '''
    try:
        response = requests.get(url)
        img = PIL.Image.open(BytesIO(response.content))
        img = img.resize((160, 160), PIL.Image.ANTIALIAS)
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        predictions = loaded_model.predict_on_batch(img_batch).flatten()
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)
        labels = ['Defective Box', 'Non Defective Box']
        return {'prediction': 'This is a {}'.format(labels[predictions.numpy()[0]])}
    except Exception as ex:
        logging.info("Error:", ex)
        exit('Error Occured: Invalid URL or Connection Error')

@app.get('/')
def index():
    return {'message': 'This is the Box Image Classification API!'}

@app.get('/predict-by-path')
def predict_path(path: str ):
    return predict_img_file(path)

@app.get('/predict-by-url')
def predict_url(url: str ):
    return predict_img_url(url)


if __name__ == '__main__':
    uvicorn.run('app:app', port=8000, host= '0.0.0.0')