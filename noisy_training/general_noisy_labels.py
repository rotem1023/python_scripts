import sys
sys.path.append('/home/dsi/rotemnizhar/dev/python_scripts')

from noisy_training.loader import get_loaders_for_training, get_num_classes
from noisy_training.file_handler import get_data_dir

import argparse
import torch
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models


# Set a random seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)

class ConvNetHam1000(nn.Module):
    def __init__(self):
        super(ConvNetHam1000, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 112 * 112, 256)  # Adjusted dimensions
        self.fc2 = nn.Linear(256, 8)  # Updated for 8 output classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 112 * 112)  # Flatten the tensor with the correct dimensions
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ConvNetMnist10(nn.Module):
    def __init__(self):
        super(ConvNetMnist10, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(4 * 56 * 56, 256)  # Adjusted dimensions
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 4 * 56 * 56)  # Flatten the tensor with the correct dimensions
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ConvNetPathMnist(nn.Module):
    def __init__(self):
        super(ConvNetPathMnist, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 256)  # Adjust the dimensions based on your input size
        self.fc2 = nn.Linear(256, 9)  # Assuming 9 output classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 14 * 14)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x    


class ConvNetCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNetCifar10, self).__init__()
        
        # Convolutional Layer Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 56 * 56, 256)  # Assuming input images are 224x224
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Convolutional Layers with Pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten the tensor before the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
# def get_model(data_set):
#     if data_set == "ham10000":
#         # return ConvNetHam1000()
#         model_ft = models.resnet18(pretrained=True)
#         set_parameter_requires_grad(model_ft, False)
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, 8)
#         return model_ft
#     elif data_set == "mnist-10":
#         return ConvNetMnist10()
#     elif data_set == "path_mnist":
#         return ConvNetPathMnist()
#     elif data_set == "cifar-10":
#         return ConvNetCifar10()
#     else:
#         raise Exception("Unsupported dataset")
    
    
def get_model(model_name, num_classes, feature_extract = True, use_pretrained=True):
    model_ft = models.resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft
    
# def get_loaders(data_set):
#     if data_set == "ham10000":
#         return get_loaders_ham
#     elif data_set == "mnist-10":
#         return get_loaders_mnist
#     elif data_set == "path_mnist":
#         return get_loaders_path_mnist
#     elif data_set == "cifar-10":
#         return get_loaders_cifar
#     else:
#         raise Exception("Unsupported dataset")
    
def _normalize_targets(targets):
    if targets.ndim > 1:
        targets = targets.squeeze()
    return targets

def create_combine_loader(train_loader, valid_loader, test_loader, batch_size):
    # Extract datasets from DataLoaders
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset
    test_dataset = test_loader.dataset

    # Combine the datasets
    combined_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])

    # Create a new DataLoader with shuffling
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return combined_loader


def calculate_accuracy(loader, model, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            targets = _normalize_targets(targets)
            data, targets = data.to(device), targets.to(device)
            # data = data.view(data.size(0), -1)  # Flatten if needed
            outputs = model(data)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return 100 * correct / total



def predict_and_evaluate(loader, model, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in loader:
            targets = _normalize_targets(targets=targets)
            data, targets = data.to(device), targets.to(device)
            # data = data.view(data.size(0), -1)  # Flatten if needed
            outputs = model(data)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            all_predictions.append(predicted.cpu())
            all_targets.append(targets.cpu())
    
    # Accuracy calculation
    accuracy = 100 * correct / total
    
    # Flatten lists into tensors
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    print(f"got {len(all_predictions)} predictions")
    return all_predictions, all_targets, accuracy

def train(dataset, model, combined_loader, device, train_loader, valid_loader, test_loader, accuracy_thresholds):
    train_accuracy_thresholds = [acc for acc in accuracy_thresholds]
    valid_accuracy_thresholds = [acc for acc in accuracy_thresholds]
    test_accuracy_thresholds = [acc for acc in accuracy_thresholds]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(combined_loader):
            # Flatten the input data if needed (e.g., for MNIST)
            # data = data.view(data.size(0), -1)

            # Move data to device if using GPU
            targets = _normalize_targets(targets=targets)
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            
            # Evaluate on train, valid, and test loaders
            if (batch_idx%1200 == 0 and epoch >19):
                save_file_if_needed(set_type='train', dataset=dataset, model=model, loader=train_loader, device=device, accs=train_accuracy_thresholds)
                save_file_if_needed(set_type='valid', dataset=dataset, model=model, loader=val_loader, device=device, accs=valid_accuracy_thresholds)
                save_file_if_needed(set_type='test', dataset=dataset, model=model, loader=test_loader, device=device, accs=test_accuracy_thresholds)    
                if len(train_accuracy_thresholds) ==0 and len(valid_accuracy_thresholds) ==0 and len(test_accuracy_thresholds) ==0:
                        print(f"finish creating noisy labels")            
                        return
            # Switch back to training mode
            model.train()
            
            print(f"Ecpoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}")
            


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")
    

def save_file_if_needed(set_type, dataset, model, loader, device, accs):
    if (len(accs)> 0):
        accuracy = calculate_accuracy(loader, model, device)
        accuracy_threshold = accs[0]
        print(f"cur acc for dataset: {dataset}, set:{set_type} is {round(accuracy,2)}, acc threshold: {accuracy_threshold}")
        if accuracy>= accuracy_threshold:
            print(f"saiving predictions: Accuracy threshold met! for accuracy threshold {accuracy_threshold} and set: {set_type}, daataset: {dataset}")
            accs.pop(0)
            save_noist_labels_for_set(set_type, dataset, accuracy_threshold, model, loader, device)

def save_noist_labels_for_set(set_type, dataset, accuracy_threshold, model, loader, device):
    predictions, tarin_labels, accuracy = predict_and_evaluate(model=model, loader=loader, device=device)
    outputdir = get_data_dir(dataset=dataset)
    os.makedirs(outputdir, exist_ok=True)
    np.save(f"{outputdir}/noisy_labels_{set_type}_dataset_{dataset}_want_acc_{accuracy_threshold}_real_acc_{round(accuracy)}", predictions.numpy())
    print(f"saved results for data set: {dataset}, set: {set_type}, wanted accuracy: {accuracy_threshold}")


def save_noiys_labels(dataset, accuracy_threshold, model, train_loader_for_pred, val_loader_for_pred, test_loader_for_pred, device):
    train_predictions, tarin_labels, train_accuracy = predict_and_evaluate(model=model, loader=train_loader_for_pred, device=device)
    valid_predictions, valid_labels, valid_accuracy = predict_and_evaluate(model=model, loader=val_loader_for_pred, device=device)
    test_predictions, test_labels, test_accuracy = predict_and_evaluate(model=model, loader=test_loader_for_pred, device=device)
    
    
    outputdir = get_data_dir(dataset=dataset)
    os.makedirs(outputdir, exist_ok=True)
    np.save(f"{outputdir}/noisy_labels_train_dataset_{dataset}_want_acc_{accuracy_threshold}_real_acc_{round(train_accuracy)}", train_predictions.numpy())
    np.save(f"{outputdir}/noisy_labels_valid_dataset_{dataset}_want_acc_{accuracy_threshold}_real_acc_{round(valid_accuracy)}", valid_predictions.numpy())
    np.save(f"{outputdir}/noisy_labels_test_dataset_{dataset}_want_acc_{accuracy_threshold}_real_acc_{round(test_accuracy)}", test_predictions.numpy())
    print(f"Create noisy labels for data set: {dataset}, accuracy threshold: {accuracy_threshold}")
    print(f"saved results for data set: {dataset}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create noisy labels')
    parser.add_argument('--dataset',
                        default='pathmnist',
                        help='data set under test',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=20,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)

    
    parser.add_argument('--acc',
                        default=80,
                        help='desired accuracy of noisy labels',
                        type=int)
    
    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    accuracy_threshold = args.acc
    dataset = args.dataset
    num_classes= get_num_classes(dataset=dataset)

    print(f"Create noisy labels for data set: {dataset}, accuracy threshold: {accuracy_threshold}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"we use device: {device}")
    model = get_model(dataset, num_classes=num_classes)

    model = model.to(device)
    train_loader,train_loader_at_eval, val_loader, test_loader = get_loaders_for_training(dataset_name=dataset)
    _,train_loader_for_pred, val_loader_for_pred, test_loader_for_pred = get_loaders_for_training(dataset_name=dataset)

    combined_loader = create_combine_loader(train_loader=train_loader_at_eval, valid_loader=val_loader, test_loader=test_loader, batch_size=batch_size)
    
    train(dataset= dataset, model=model, combined_loader=combined_loader, device=device,
          train_loader=train_loader_for_pred, valid_loader=val_loader_for_pred, test_loader=test_loader_for_pred, accuracy_thresholds=[95])
    




