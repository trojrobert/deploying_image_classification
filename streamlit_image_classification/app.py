import requests

import streamlit as st

from PIL import Image
import torch
from torchvision import models, transforms 

def get_prediction(input_image, model, labels):
    """[summary]

    Args:
        image (): input image
        model ([type]): pytorch model 
        labels([type]): ImageNet readable labels

    Returns:
        [type]: Predictions from the labels
    """

    tensor = transform_image(image=input_image)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    y_hat = y_hat.numpy()

    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(tensor)[0], dim=0)
    
    return labels[y_hat[0]]


def transform_image(image):
    """ Transform image to fit model

    Args:
        image (image): Input image from the user

    Returns:
        tensor: transformed image 
    """
    transformation = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return transformation(image).unsqueeze(0)


@st.cache
def load_model_and_labels():
    # Make sure to pass `pretrained` as `True` to use the pretrained weights:
    # Since we are using our model only for inference, switch to `eval` mode:
    model = models.resnet18(pretrained=True).eval()

    #download human-readable label for ImageNet
    response = requests.get("https://git.io/JJkYN")
    labels = response.text.split("\n")
    return model, labels


def main():
    
    st.title("Predict objects in an image")
    st.write("This application can predict objests in an image , but works best when only one object is in the image")

    model, labels = load_model_and_labels()
    
    #Create a UI component to read image
    image_file  = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if image_file:
        
        #divide your interface to two parts
        left_column, right_column = st.columns(2)
        left_column.image(image_file, caption="Uploaded image", use_column_width=True)
        input_image = Image.open(image_file)

        #create a UI component to create a button
        pred_button = st.button("Predict")
        
        
        if pred_button:

            prediction = get_prediction(input_image, model, labels)
            right_column.title("Prediction")
            right_column.write(prediction)

if __name__ == '__main__':
    main()