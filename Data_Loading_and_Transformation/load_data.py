# src/data/load_data.py

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_transforms():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def load_datasets(train_dir, valid_dir, transform, batch_size=32):
    trainset = datasets.ImageFolder(root=train_dir, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.ImageFolder(root=valid_dir, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainset, trainloader, testset, testloader
