import requests

import gradio as gr

from PIL import Image
import torch
from torchvision import models, transforms 

def get_prediction(input_image, model, labels):
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
    transformation = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return transformation(image).unsqueeze(0)


def load_model_and_labels():
    # Make sure to pass `pretrained` as `True` to use the pretrained weights:
    # Since we are using our model only for inference, switch to `eval` mode:
    model = models.resnet18(pretrained=True).eval()

    # Download human-readable labels for ImageNet.
    response = requests.get("https://git.io/JJkYN")
    labels = response.text.split("\n")
   
    return model, labels


def main(input_image):
    
    model, labels = load_model_and_labels()
    #image = image.reshape(1, -1)

    prediction = get_prediction(input_image, model, labels)
    return prediction

iface = gr.Interface(
    fn=main,
    inputs=gr.inputs.Image(shape=(224, 224)),
    outputs=gr.outputs.Label(num_top_classes=3),
    title="Predict objects in an image",
    description="This application can predict objects in an image , but works best when only one object is in the image",
    live=True,
    interpretation="default",
    capture_session=True
)


if __name__ == '__main__':
    iface.launch(share=True)