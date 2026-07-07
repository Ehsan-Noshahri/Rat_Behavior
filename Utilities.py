import numpy as np
import torch
import torchvision
import time
import matplotlib.pyplot as plt

import matplotlib
import sns
import os
import json
import skimage

from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.transforms import v2
from sklearn.utils import shuffle
import torch.nn.functional as F
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom Dataset class with transformations
class MyDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        image = self.x_data[idx]
        label = self.y_data[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

# # New Train

import os
import time
import json
import torch
from tempfile import TemporaryDirectory
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, early_stopping_patience=20, save_dir="output"):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare directory to save model and history
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    history_path = os.path.join(save_dir, 'training_history.json')

    # Initialize training history
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}

    since = time.time()
    best_acc = 0.0
    early_stop_count = 0  # Initialize early stopping counter

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # Check if the model is in training mode and auxiliary logits are enabled
                    if isinstance(outputs, tuple):  
                        # Extract the main output (first element of the tuple)
                        outputs = outputs[0]
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Compute epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save metrics in history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_accuracy'].append(epoch_acc.item())
                history['lr'].append(optimizer.param_groups[0]['lr'])
            else:
                history['val_loss'].append(epoch_loss)
                history['val_accuracy'].append(epoch_acc.item())

            

            # Adjust learning rate based on loss (only for ReduceLROnPlateau)
            prev_lr = [group['lr'] for group in optimizer.param_groups]
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if phase == 'val': scheduler.step(epoch_loss)
            elif phase == 'train': scheduler.step()  # safe for other schedulers like StepLR
            
            new_lr = [group['lr'] for group in optimizer.param_groups]
            
            if new_lr != prev_lr: 
                prev_str = ', '.join(f"{lr:.2e}" for lr in prev_lr)
                new_str  = ', '.join(f"{lr:.2e}" for lr in new_lr)
                print(f"📉 LR changed: {prev_str} → {new_str}")

                
            # Early stopping and model saving logic
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    early_stop_count = 0  # Reset early stopping if improved
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved with accuracy: {best_acc:.4f}")
                else:
                    early_stop_count += 1  # Increase counter if no improvement

        print()
        # Early stopping check
        if early_stop_count > early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Training summary
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    # Load best model weights (now correctly saved in the persistent directory)
    model.load_state_dict(torch.load(best_model_path))
    print(f"Best model loaded from {best_model_path}")

    # Save training history
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")

    return model  # Now it returns at the end after all prints


# # Evaluation Function
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score,matthews_corrcoef

# Function to evaluate and collect metrics
def evaluate(net,loader, dataset_name):
    # Save original training mode
    was_training = net.training

    net.eval()  # Set to eval mode for accurate predictions
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.view(-1).cpu().numpy())
            y_pred.extend(predicted.view(-1).cpu().numpy())

    # Restore original mode
    if was_training:
        net.train()

    d = classification_report(y_true, y_pred, output_dict=True, zero_division="warn")
        
    # "accuracy", "precision","recall","f1-score"("weighted avg"),"precision","recall","f1-score" ("macro avg")+ cohens kappa+ matthews coefficient
    metrics=[d["accuracy"]]+ [d["weighted avg"][a] for a in ["precision","recall","f1-score"]]+[d["macro avg"][a] for a in ["precision","recall","f1-score"]]+[cohen_kappa_score(y_true, y_pred)] +[matthews_corrcoef(y_true, y_pred)]
    metrics=np.array(metrics)

    # Confusion Matrix
    confusion= confusion_matrix(y_true, y_pred)
    normalized_confusion= confusion_matrix(y_true, y_pred, normalize="true")
    
    print(f"{dataset_name} Accuracy: {metrics[0]}")
    return metrics, confusion, normalized_confusion 



# # Model Loaders
from torchvision import models

def load_resnet(model_index):
    # ResNet
    net = [models.resnet18(weights='IMAGENET1K_V1'), models.resnet50(weights='IMAGENET1K_V1'), models.resnet152(weights='IMAGENET1K_V1')] [model_index]
    net.fc = nn.Linear(in_features=net.fc.in_features, out_features=len(class_list))
    return net

def load_efficientnet(model_index):
    # EfficientNet
    net = [models.efficientnet_v2_s(weights='IMAGENET1K_V1'), models.efficientnet_v2_m(weights='IMAGENET1K_V1'), models.efficientnet_v2_l(weights='IMAGENET1K_V1')][model_index]
    net.classifier[1] = nn.Linear(in_features=net.classifier[1].in_features, out_features=len(class_list))
    return net

def load_densenet(model_index):
    # EfficientNet
    net = [models.densenet121(weights='IMAGENET1K_V1'), models.densenet169(weights='IMAGENET1K_V1'), models.densenet201(weights='IMAGENET1K_V1')][model_index]
    net.classifier = nn.Linear(in_features=net.classifier.in_features, out_features=len(class_list))
    return net 

def load_inceptionnet(model_index):
    # InceptionNet
    net = models.inception_v3(weights='IMAGENET1K_V1')
    net.fc = nn.Linear(in_features=net.fc.in_features, out_features=len(class_list))
    net.AuxLogits.fc = nn.Linear(in_features=net.AuxLogits.fc.in_features, out_features=len(class_list))
    
    return net