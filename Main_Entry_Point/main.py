# src/main.py

import torch
import torch.optim as optim
import torch.nn as nn
from src.data.load_data import get_transforms, load_datasets
from src.models.model import get_squeezenet
from src.models.train import train_model
from src.utils.evaluation import evaluate_model
from src.utils.helpers import plot_training_metrics
import matplotlib.pyplot as plt

def main():
    # Configuration
    train_dir = '/home/ec/Downloads/New Plant Diseases Dataset(Augmented)/train/'
    valid_dir = '/home/ec/Downloads/New Plant Diseases Dataset(Augmented)/valid/'
    model_save_path = '/home/ec/Downloads/squeezenet1_1_plant_disease.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.001

    # Data Preparation
    transform = get_transforms()
    trainset, trainloader, testset, testloader = load_datasets(train_dir, valid_dir, transform, batch_size)

    # Model Setup
    num_classes = len(trainset.classes)
    model = get_squeezenet(num_classes, pretrained=False)

    # Move model to device
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train_losses, train_accuracies = train_model(model, trainloader, criterion, optimizer, device, num_epochs)

    # Plot Training Metrics
    epochs = range(1, num_epochs + 1)
    plot_training_metrics(epochs, train_losses, train_accuracies)

    # Save the Trained Model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Evaluation
    model.load_state_dict(torch.load(model_save_path))
    accuracy, conf_matrix, report = evaluate_model(model, testloader, device, trainset.classes)

if __name__ == '__main__':
    main()
