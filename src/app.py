import streamlit as st
import pandas as pd
import  numpy as np

from PIL import Image
from torchvision import models

from preprocess import  preprocessing_image
from inference import  run_inference, get_prediction_class
from utils import read_imagenet_classnames

def run():

    model = models.resnet18(pretrained=True)
    st.title("Image Classification with Machine Learning")

    image_file  = st.file_uploader("Upload an image")

    imagenet_classes = read_imagenet_classnames("./data/imagenet_classnames.txt")


    if image_file:
        st.image(image_file, use_column_width=True)
        image = Image.open(image_file)
        pred_button = st.button("Predict")
        
        if pred_button:
            processed_img = preprocessing_image(image)
            predictions = run_inference(model, processed_img)
            
            prediction_classes = get_prediction_class(predictions, imagenet_classes)
            st.write(prediction_classes)

if __name__ == '__main__':
    run()