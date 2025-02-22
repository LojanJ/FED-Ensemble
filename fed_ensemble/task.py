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
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.metrics import precision_score, recall_score, f1_score

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, config):
        super(Net, self).__init__()
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


fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int, batch_size = 32):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)

        fds = FederatedDataset(
            dataset="mnist",
            partitioners= {"train":partitioner}
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test3
    partition_train_test = partition.train_test_split(test_size=0.4, seed=42)

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(partition_train_test["test"], batch_size=batch_size)

    return trainloader, val_loader

def apply_transforms(batch):
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5), (0.5))]
    )

    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0


    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


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

def save_checkpoint(epoch, models, optimizers, filename):

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create state dicts for all models and optimizers
    models_state = {name: model.state_dict() for name, model in models.items()}
    optimizers_state = {name: opt.state_dict() for name, opt in optimizers.items()}

    checkpoint = {
        'epoch': epoch,
        'models': models_state,
        'optimizers': optimizers_state
    }
    torch.save(checkpoint, filename)


def load_config(config_path):
    """Load the configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config