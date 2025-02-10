import json
import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fed_ensemble.task import Net, get_weights, load_config, load_data, set_weights, test, train, compute_features

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, node_id, net, trainloader, valloader, local_epochs):
        self.net = net
        self.node_id = node_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        try:
            # Set weights and verify they were set correctly
            set_weights(self.net, parameters)
            
            # Verify data loading
            if len(self.trainloader.dataset) == 0:
                raise ValueError("Training dataset is empty")
                
            # Add training progress logging3
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
            ) 
            # Compute features with verification
            local_features, local_labels = compute_features(self.net, self.trainloader, self.device)
            if local_features.size == 0 or local_labels.size == 0:
                raise ValueError("Feature computation returned empty arrays")
            
        
            if isinstance(local_features, np.ndarray):
                local_features = local_features.tolist()
                
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {
                    "train_loss": float(train_loss),  # Ensure loss is a Python float
                    "local_features": json.dumps(local_features),
                    "local_labels": json.dumps(local_labels.tolist()),
                    'node_id': self.node_id
                },
            )
        except Exception as e:

            
            (f"Error in client fit: {e}")
                # Return minimal valid response
            return (
                get_weights(self.net),  # Return original weights
                len(self.trainloader.dataset),
                {"train_loss": 0.0, "local_features": [], "local_labels": []},
            )
        
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        evaluate = test(self.net, self.valloader, self.device)
        return evaluate['loss'], len(self.valloader.dataset), {"accuracy": evaluate["accuracy"], 
                                                               "loss": evaluate["loss"],
                                                               "f1": evaluate['f1_score'],
                                                               "recall": evaluate['recall'],
                                                               "precision": evaluate['precision']}
    

class MaliciousClient(FlowerClient):
    def __init__(self, node_id, net, trainloader, valloader, local_epochs, attack_type = "label_flip"):
        super().__init__(node_id, net, trainloader, valloader, local_epochs)
        self.attack_type = attack_type
        self.poison_frac = 0.3

    def _poison_data(self, features : torch.Tensor, labels : torch.Tensor):
        poison_size = int(len(features) * self.poison_frac)
        indices = np.random.choice(len(features), poison_size, replace=False)

        poisoned_features  = features.clone()
        poisoned_labels = labels.clone()

        if self.attack_type == "label_flip":
            # Flip labels to incorrect classes
            poisoned_labels[indices] = (labels[indices].to(self.device) + 5) % 10

        elif self.attack_type == "backdoor":
            # Add a backdoor trigger (pattern) to images
            trigger_pattern = torch.ones((1, 1, 4, 4)) * 0.5
            poisoned_features[indices, :, -4:, -4:] = trigger_pattern.to(self.device)
            poisoned_labels[indices] = 9  # Target label
            
        elif self.attack_type == "gradient_ascent":
            # Perform gradient ascent instead of descent
            return features, labels, True
    
        return poisoned_features, poisoned_labels, False
    
    def _generate_malicious_features(self, features: np.ndarray) -> np.ndarray:
        # Create anomalous features to evade detection
        noise = np.random.normal(0, 0.1, features.shape)
        return features + noise
    
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        self.net.train()
        
        # Training with poisoned data
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters())
        
        running_loss = 0.0
        for batch in self.trainloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Apply poisoning
            poisoned_images, poisoned_labels, grad_attack = self._poison_data(images, labels)
            
            optimizer.zero_grad()
            outputs = self.net(poisoned_images)
            loss = criterion(outputs, poisoned_labels)
            
            if grad_attack:
                loss = -loss  # Gradient ascent for model disruption
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Generate malicious features for server
        features, labels = compute_features(self.net, self.trainloader, self.device)
        malicious_features = self._generate_malicious_features(features)
        
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": float(running_loss / len(self.trainloader)),
                "local_features": json.dumps(malicious_features.tolist()),
                "local_labels": json.dumps(labels.tolist()),
                'node_id': self.node_id
            }
        )

def client_fn(context: Context):
    # Load from config
    config = load_config('fed_ensemble/config.json')
    local_epochs = config['local-epochs']
    node_id = context.node_config['partition-id']

    train_loader, val_loader = load_data(
        partition_id=node_id, 
        num_partitions=context.node_config['num-partitions']
    )
    net = Net(config)

    malicious_partition_ids = config['malicious_clients_id']

    if node_id in malicious_partition_ids:
        attack_type = "gradient_ascent"  
        return MaliciousClient(
            net=net,
            node_id = node_id,
            trainloader=train_loader,
            valloader=val_loader,
            attack_type=attack_type,
            local_epochs=local_epochs
        ).to_client()
    else:
        return FlowerClient(
            net=net,
            node_id = node_id,
            trainloader=train_loader,
            valloader=val_loader,
            local_epochs=local_epochs
        ).to_client()

# Create ClientApp
app = ClientApp(client_fn=client_fn)