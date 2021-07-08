from PIL import Image

from torchvision import transforms 


transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    
])

def preprocessing_image(image):
    """process the image

    Args:
        image ([type]): [description]s
    """

    trans_image = transformations(image)
  
    return trans_image