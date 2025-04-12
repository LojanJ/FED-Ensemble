from collections import OrderedDict

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, ColorJitter
from sklearn.metrics import precision_score, recall_score, f1_score

class MnistNet(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blidtz')"""
    def __init__(self, config):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(config['num_channels'], 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def feature_extractor(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class CifarNet(nn.Module):
    """Obtained codebase from https://github.com/nikosgalanis/data-poisoning-defense-fl/blob/main/DefenseFederated/src/models.py """
    def __init__(self, config):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)  
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.1)


        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout_fc1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.dropout_fc2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 10)

        # Additional initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')

    def feature_extractor(self, x):
        # Original spatial reduction with enhanced channels
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), 0.01)) 
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), 0.01)) 
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), 0.01)) 
        x = self.dropout_conv(x)
        x = x.view(-1, 256 * 4 * 4)  # 256*4*4 = 4096
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = self.dropout_fc1(x)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
fds = None

def load_data(partition_id: int, num_partitions: int, dataset: str, batch_size =  32):
    global fds

    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        if dataset == 'mnist':
            fds = FederatedDataset(
                dataset="mnist",
                partitioners= {"train":partitioner}
            )
        elif dataset == 'cifar':
            fds = FederatedDataset(
                dataset="cifar10",
                partitioners= {"train":partitioner}
            )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 60% train, 40% test
    partition_train_test = partition.train_test_split(test_size=0.4, seed=42)
    print(f"Node {partition_id} has {len(partition_train_test['train'])} training samples and {len(partition_train_test['test'])} test samples.")

    partition_train_test = partition_train_test.with_transform(
                            lambda batch: apply_transforms(batch, dataset))

    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(partition_train_test["test"], batch_size=batch_size)

    return trainloader, val_loader

def apply_transforms(batch, dataset):
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5), (0.5))]
        if dataset == 'mnist' else
        [
            ToTensor(),
            RandomHorizontalFlip(),
            RandomCrop(32, padding=4),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    # Handle different dataset keys and standardize
    if 'img' in batch:
        images = batch.pop('img')
        batch['image'] = [pytorch_transforms(img) for img in images]
    elif 'image' in batch:
        batch['image'] = [pytorch_transforms(img) for img in batch['image']]
    
    # Ensure label key is consistent
    if 'label' not in batch and 'labels' in batch:
        batch['label'] = batch.pop('labels')
    return batch

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_trainloss = running_loss / len(trainloader)
    accuracy = correct / total  # as percentage
    
    return avg_trainloss, accuracy

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {
            "loss" : loss / len(testloader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
            }
    
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def compute_features(net, dataloader, device):
    net.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            # Extract features from the last conv layer
            x = net.feature_extractor(images)
            features.append(x.cpu().numpy())
    return np.concatenate(features, axis=0)

def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.json')
    """Load the configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config['dataset'] == 'mnist':
        config['num_channels'] = 1
        config['input_dim'] = 256 
    elif config['dataset'] == 'cifar':
        config['num_channels'] = 3
        config['input_dim'] = 4096
    return config
