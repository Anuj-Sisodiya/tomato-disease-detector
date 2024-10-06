# src/models/model.py

import torch.nn as nn
import torchvision.models as models

def get_squeezenet(num_classes, pretrained=False):
    model = models.squeezenet1_1(pretrained=pretrained)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    return model
