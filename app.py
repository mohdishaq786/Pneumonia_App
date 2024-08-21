import streamlit as st
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage.io import imread
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model 
import requests
from io import BytesIO

# Output file path
output_path = 'model.h5'

# URL of the file to download
url = 'https://pixel-go-app.apyhi.com/seg_test/pnue.h5'

response = requests.get(url)

if response.status_code == 200:
    # Open a file in write-binary mode and save the content
    with open(output_path, 'wb') as file:
        file.write(response.content)
    print("File downloaded successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")

model = load_model('model.h5')
  
st.title('Pneumonia Detector')
st.image('https://upload.wikimedia.org/wikipedia/commons/8/81/Chest_radiograph_in_influensa_and_H_influenzae%2C_posteroanterior%2C_annotated.jpg', width = 500)

def process(image):
    if image is not None:
      if type(image) != str:
        img = np.array(Image.open(image).convert('RGB'))
        img = img/255
        img = resize(img, (256, 256))
        st.image(img, caption = 'Input', width = 256)
        return img
      else:
        response = requests.get(image)
        img = np.array(Image.open(BytesIO(response.content).convert('RGB')))
        img = img/255
        img = resize(img, (256, 256))
        st.image(img, caption = 'Input', width = 256)
        return img
    else:
        pass #st.text('Upload a Image')	
 
def predict(image):
    img = np.expand_dims(image, axis = 0)
    pred = model.predict(img)[0]
    prob = pred
    pred = int(pred>=0.9)
        
    if pred == 1:
        class_ = 'Pneumonia'
        
    else:
        class_ = 'Normal'
        
    st.title('Result: ' + class_)

    if class_ == 'Pneumonia':
      st.title('Probality: ' + str(prob[0]))
      st.title('Kindly contact the doctor')
    else:
      st.title('Probality: ' + str(1-prob[0]))
      st.title('Well and good')
    
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
except:
  pass

try:
  predict(img)
except:
  pass
