

import torch

from torch import cuda, device


def run_inference(model, processed_image, top_predictions = 10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    image = processed_image.to(device)
    image = torch.unsqueeze(image, 0)
    predictions = model(image)
    probabilities = torch.softmax(predictions, 1)
    pred_indices = torch.argsort(probabilities, 1,\
                        descending=True)[:, : top_predictions]
    probabilities = torch.gather(probabilities, 1, pred_indices)
    probabilities = (probabilities * 100).detach().numpy()
    pred_indices = pred_indices.detach().numpy()
 
    return probabilities, pred_indices


def get_prediction_class(predictions, imagenet_classes):
    
    probabilities, pred_indices = predictions
    # print(f'probabilities {probabilities[0]}')
    # print(f'pred_indices {pred_indices[0]}')
    # print(f'{len(probabilities[0])}')
    prediction_classes = [f"{imagenet_classes[pred_indices[0][j]][0]} ({100 * probabilities[0][j]:.2f}%)" \
                for j in range(len(probabilities[0]))]
    
    return prediction_classes