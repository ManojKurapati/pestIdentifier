# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

# TensorFlow and tf.keras
import json
import tensorflow as tf
#import tensorflow_hub as hub
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
from PIL import Image, ImageOps
class_names = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']
#print(class_names)
img_height = 224
img_width = 224
#load model
import streamlit as st

# Model saved with Keras model.save()
#MODEL_PATH = 'saved_model/flw_classifier_model.h5'
MODEL_PATH = 'saved_model'


# Load your own trained model
st.cache()
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_model()
# model._make_predict_function()          # Necessary 
print('Model loaded. Start serving...')

import os

#from classification import model_predict


st.set_page_config(layout="wide")
st.title("Farmers Enemies")
html_temp = """

  <div style="background-color:blue;padding:10px">
  <h2 style="color:grey;text-align:center;">Streamlit App </h2>
  </div>
  
  """
st.header("Aggricultural Pests Classification")
st.image(os.path.join('images','planttomato.jpg'), use_column_width  = True)
with st.expander("About"):

        st.write("The Model is trained on dataset contains plant Pests.")
                 

st.text("Upload Image for pest classification")
st.sidebar.header('List of Pests that app can recognize')
st.sidebar.header('Plants Pests')
st.sidebar.text("aphids")
st.sidebar.text("armyworm")
st.sidebar.text("beetle")
st.sidebar.text("bollworm")
st.sidebar.text("grasshopper")
st.sidebar.text("mites")
st.sidebar.text("sawfly")
st.sidebar.text("stem_borer")

#dpreprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
uploaded_file = st.file_uploader("Choose a image ...", type="jpg")
if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption='Uploaded image.', use_column_width=True)
        st.write("")
        st.write("Detecting Pest...")    
        image1 = image1.resize((img_height, img_width))
        #img_array = image.img_to_array(image1)
        #img_array = tf.expand_dims(img_array, 0) # Create a ba
        #img_array = preprocess_input(img_array)
        img_array = np.array(image1)/255.0
        predictions = model.predict(img_array[np.newaxis])
        #predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) 
        pred_proba = "{:.3f}".format(100*np.max(score))  # Max probability
        pred_class = class_names[np.argmax(score)]   # ImageNet Decode
        #pred_class = decode_predictions(score, top=1)
        result = str(pred_class)               
        result = result.capitalize()
        st.text("Predicted Image is belong to...")
        st.write(result)
        st.text("Probabitity of Predicted Image is...")
        st.write(pred_proba)
               
     
        
