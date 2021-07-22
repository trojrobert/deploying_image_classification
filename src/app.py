import os

import streamlit as st
import pandas as pd
import  numpy as np

from PIL import Image
from torchvision import models

from preprocess import  preprocess_image
from inference import  run_inference, get_prediction_class
from utils import read_imagenet_classnames

IMAGE_DISPLAY_SIZE = (330, 330)



def run():
    
    model = models.resnet18(pretrained=True)
    st.title("Predict objects in an image")
    st.write("This application knows the objects in an image , but works best when only one object is in the image")

    image_file  = st.file_uploader("Upload an image")

    imagenet_classes = read_imagenet_classnames(f"{os.getcwd()}/data/imagenet_classnames.txt")


    if image_file:
       
        left_column, right_column = st.beta_columns(2)
        left_column.image(image_file, caption="Uploaded image", use_column_width=True)
        image = Image.open(image_file)
        pred_button = st.button("Predict")
        
        
        if pred_button:
            processed_img = preprocess_image(image)
            predictions = run_inference(model, processed_img)
            
            prediction_classes = get_prediction_class(predictions, imagenet_classes)
            
            for prediction in prediction_classes:
                right_column.write(prediction)

if __name__ == '__main__':
    run()