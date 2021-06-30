from keras.preprocessing import image
import streamlit as st
from config import *
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os

from db import AIModel, Image
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import eli5

# you may want to keep logging enabled when doing your own work
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR) # disable Tensorflow warnings for this tutorial
import warnings
warnings.simplefilter("ignore") # disable Keras warnings for this tutorial
import keras
from keras.applications import mobilenet_v2,resnet50,DenseNet121

def opendb():
    engine = create_engine('sqlite:///db.sqlite3') # connect
    Session =  sessionmaker(bind=engine)
    return Session()

def save_file(file,path):
    try:
        db = opendb()
        ext = file.type.split('/')[1] # second piece
        img = Image(filename=file.name,extension=ext,filepath=path)
        db.add(img)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.write("database error:",e)
        return False

st.sidebar.header(PROJECT_NAME)

choice = st.sidebar.radio("Project Menu", MENU_OPTION)

if choice == 'Image Upload':
    st.title("Upload Image")
    files = st.file_uploader("select a image",type=['jpg','png'],accept_multiple_files=True)
    
    if files:
        for file in files: 
            path = os.path.join('uploads',file.name)
            with open(path,'wb') as f:
                f.write(file.getbuffer())
                status = save_file(file,path)
                if status:
                    st.sidebar.success("file uploaded")
                    st.sidebar.image(path,use_column_width=True)
                else:
                    st.sidebar.error('upload failed')

if choice == 'Model Upload':
    dims = (224,224)
    modellist = ['mobilenet_v2','resnet50','desnseNet121']
    st.title("Upload Model")
    modelname = st.selectbox("select an ai model",modellist)
    if modelname == modellist[0]:
        model = mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet', classes=1000)
        dims = model.input_shape[1:3] 
    elif modelname ==modellist[1]:
        model = resnet50.ResNet50(weights='imagenet')
        dims = model.input_shape[1:3] 
        
    elif modelname ==modellist[2]:
        model = DenseNet121()
        dims = model.input_shape[1:3] 

    st.subheader(f"selected model is {modelname}")
    st.text(f'image shape will be {dims}')
    images = opendb().query(Image).all()
    img = st.selectbox("choose image ",images)
    if img and st.button("view what the model is looking at"):
        st.image(img.filepath)
        im = keras.preprocessing.image.load_img(img.filepath, target_size=dims) # -> PIL image
        doc = keras.preprocessing.image.img_to_array(im)
        doc = np.expand_dims(doc, axis=0)
        predictions = model.predict(doc)
        st.write(type(predictions), predictions.shape)
        if modelname==modellist[0]:
            top = mobilenet_v2.decode_predictions(predictions)
        elif modelname ==modellist[1]:
            top = resnet50.decode_predictions(predictions)
        top_indices = np.argsort(predictions)[0, ::-1][:5]
        st.title("output")
        st.subheader("detected object and confidence")
        st.image(eli5.show_prediction(model, doc))
        try:st.write(top)
        except:pass

if choice == 'About project':
    st.title("What is the project")

    st.image('Grad Cam.png', use_column_width=True)
    st.write('A technique for making Convolutional Neural Network (CNN)-based models more transparent by visualizing the regions of input that are “important” for predictions from these models — or visual explanations. Using Grad-CAM, we can visually validate where our network is looking, verifying that it is indeed looking at the correct patterns in the image and activating around those patterns')

    st.title("Project Info")
    st.write('Instead of the GAP, Grad CAM get the gradient value for the weight. Grad-CAM does not need to rid off the FC layer. Furthermore, it can be applied for every task where CAM could be only applied on the Classification')

    st.title("Objective")
    st.write('It is artificial intelligent based application. Through this application scientists and engineer can know what AI think in background and useful to explain what AI think, predict and why.')

if choice == 'Creator info':
    st.title("About the Project creators")
    
    st.image('project image.jpg', use_column_width=True)

    st.title('Mohd Arshad Khan')
    st.write('Myself Mohd Arshad Khan from Piprauli Bazar Gorakhpur(UP). I am pursuing BCA from Babu Banarasi Das University Lucknow. Im looking forward to MCA.')

    st.title('Rahul Yadav')
    st.write('Myself Rahul Yadav from Pitambara Vihar Colony Matiyari Lucknow (UP). I am pursuing BCA from Babu Banarasi Das University Lucknow. Im looking forward to MCA')