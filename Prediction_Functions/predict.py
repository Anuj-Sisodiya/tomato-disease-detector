# src/predict.py

import torch
from PIL import Image
import os
from torchvision import transforms
import torch.nn.functional as F

def predict_images(image_paths, model, transform, classes, device, top_k=5):
    model.to(device)
    model.eval()
    predictions = []
    all_predicted_labels = []
    all_probabilities = []

    for image_path in image_paths:
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probabilities = F.softmax(outputs, dim=1)
                top_probs, top_classes = probabilities.topk(top_k, dim=1)

                _, predicted = torch.max(outputs, 1)
                predicted_class = classes[predicted.item()]
                predicted_probability = probabilities[0][predicted.item()].item()

                predictions.append((image_path, predicted_class, predicted_probability,
                                    [classes[cls] for cls in top_classes.cpu().numpy()[0]],
                                    top_probs.cpu().numpy()[0]))
                all_predicted_labels.append(predicted.item())
                all_probabilities.append(probabilities.cpu().numpy())
        else:
            print(f'File does not exist: {image_path}')
            predictions.append((image_path, 'File not found', 0, [], []))

    return predictions, all_predicted_labels, all_probabilities

def display_predictions(predictions):
    for path, prediction, probability, top_classes, top_probs in predictions:
        print(f'The disease in the image {path} is: {prediction} with probability {probability:.4f}')
        for cls, prob in zip(top_classes, top_probs):
            print(f'  Top class: {cls} with probability {prob:.4f}')
        print('-' * 50)
