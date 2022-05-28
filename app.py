import streamlit as st
from skimage.transform import resize
from skimage.color import gray2rgb
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model 
model = load_model('model.h5')
  
st.title('Pneumonia Detector')

img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def process(image):
    if image is not None:
        img = np.array(Image.open(image))
        img = img/255
        img = resize(img, (256, 256))
        return img
    else:
        st.text('Upload a Image')	
if img_file is not None:
    img = process(img_file)

    st.image(img, clamp = True, channels = 'RGB', caption = 'Input Image')

def predict(image):
    img = gray2rgb(image)
    img = np.expand_dims(img, axis = 0)   
    pred = model.predict_classes(img)[0]
    if pred == 1:
        class_ = 'Pneumonia'
    else:
        class_ = 'Normal'
    
    st.title('Result: ' + class_)

if img_file is not None:
    predict(img)
