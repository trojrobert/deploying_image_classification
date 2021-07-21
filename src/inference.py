

import torch

from torch import cuda, device
import torch.nn.functional as F


def run_inference(model, processed_image, top_predictions = 10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    image = processed_image.to(device)
    image = torch.unsqueeze(image, 0)
    model.eval()
    predictions = model(image)
    probabilities, pred_indices = F.softmax(predictions, 1).topk(top_predictions)
    probabilities = (probabilities * 100).detach().numpy()
    pred_indices = pred_indices.detach().numpy()
 
    return probabilities, pred_indices


def get_prediction_class(predictions, imagenet_classes):
    
    probabilities, pred_indices = predictions
    prediction_classes = [f"{imagenet_classes[pred_indices[0][j]][0]} ({probabilities[0][j]:.2f}%)" \
                for j in range(len(probabilities[0]))]
    
    return prediction_classes