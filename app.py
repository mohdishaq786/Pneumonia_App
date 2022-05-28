import streamlit as st
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage.io import imread
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model 
model = load_model('model.h5')
  
st.title('Pneumonia Detector')

def process(image):
    if image is not None:
      if type(image) != str:
        img = np.array(Image.open(image))
        img = img/255
        img = resize(img, (256, 256))
        return img
      else:
        img = imread(image)
        img = img/255
        img = resize(img, (256, 256))
        return img
    else:
        pass #st.text('Upload a Image')	

choice = st.selectbox('Choose one of the following', ('URL', 'Upload Image'))
try:
  if choice == 'URL':
    image_path = st.text_input('Enter image URL...')
    try:
      img = process(image_path)
    except:
      st.markdown('Enter a URL')

  if choice == 'Upload Image':
    img = st.file_uploader('Upload an Image', type=["png", "jpg", "jpeg"])
    try:
      img = process(img)
    except:
        st.markdown('Upload a valid image')

def predict(image):
    img = gray2rgb(image)
    img = np.expand_dims(img, axis = 0)   
    pred = model.predict(img)[0]
    pred = int(pred>=0.9)
    if pred == 1:
        class_ = 'Pneumonia'
    else:
        class_ = 'Normal'
    
    st.title('Result: ' + class_)

if img is not None:
    predict(img)
