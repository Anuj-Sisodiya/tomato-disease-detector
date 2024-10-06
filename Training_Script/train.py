# src/models/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train_model(model, trainloader, criterion, optimizer, device, num_epochs=30, print_every=100):
    model.to(device)
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for i, data in enumerate(loop):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update running loss
            running_loss += loss.item()

            # Update tqdm progress bar
            if (i + 1) % print_every == 0:
                loop.set_postfix(loss=running_loss / print_every,
                                 accuracy=100 * correct / total)
                running_loss = 0.0

        # Calculate average epoch training loss and accuracy
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f'Finished Epoch {epoch + 1}/{num_epochs}')

    print('Finished Training')
    return train_losses, train_accuracies
