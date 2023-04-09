# Importing necessary libraries
from pydoc import doc                  # for accessing Python documentation
import tensorflow as tf                # for building and training machine learning models
from keras.models import load_model     # for loading pre-trained machine learning models
from PIL import Image, ImageOps         # for manipulating and processing images
import numpy as np                      # for mathematical operations on arrays
import streamlit as st                  # for building web applications
import h5py                             # for accessing data from HDF5 files

#
# Opening and displaying the banner of the app
image = Image.open('doc.png')
st.image(image)
st.title("Breast Cancer Detection App")
st.header("This prediction was made using Fine Needle Aspirate Images of sample patient data from the UCI Wisconsin dataset")
st.text("Made by Karl Tortebecker (karlintchoumi@gmail.com)")

# Loading the pre-trained machine learning model
model = load_model('keras_model.h5')

# Defining a function to classify the image using the pre-trained model
def teachable_machine_classification(img, model, size=(224,224)):
    # Pre-processing the image
    data = np.ndarray(shape=(1, size[0], size[1], 3), dtype=np.float32)      # creating an empty 4D array to hold the image
    image = ImageOps.fit(img, size, Image.ANTIALIAS)                         # resizing and cropping the image to fit the model's input size
    image_array = np.asarray(image)                                         # converting the image to an array
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1    # normalizing the pixel values of the image to [-1, 1]
    data[0] = normalized_image_array                                         # filling the array with the normalized image
    prediction = model.predict(data)                                         # using the pre-trained model to make a prediction
    return prediction[0]                                                     # returning the prediction (a list of probabilities)

# Building the main application
uploaded_file = st.file_uploader("Upload the FNA biopsied image ...", type=["png", "jpg", "jpeg"])   # creating a file uploader
if uploaded_file is not None:                                                                            # checking if a file has been uploaded
    image = Image.open(uploaded_file)                                                                    # opening the uploaded image
    st.image(image, caption='Uploaded image', use_column_width=True)                                     # displaying the uploaded image
    prediction = teachable_machine_classification(image, model)                                         # using the pre-trained model to classify the image
    if prediction[0] > prediction[1]:                                                                    # checking if the image is most likely benign
        st.success("The image is most likely benign")                                                     # displaying a success message
    else:                                                                                                # otherwise, the image is most likely malignant
        st.error("The image is most likely malignant")                                                    # displaying an error message
