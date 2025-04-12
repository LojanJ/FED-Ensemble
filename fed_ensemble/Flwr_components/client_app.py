import json
import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fed_ensemble.task import MnistNet, CifarNet, get_weights, load_config, load_data, set_weights, test, train, compute_features

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
            train_loss, train_accuracy = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
            ) 
            # Compute features with verification
            local_features = compute_features(self.net, self.trainloader, self.device)
            if local_features.size == 0:
                raise ValueError("Feature computation returned empty arrays")
            
        
            if isinstance(local_features, np.ndarray):
                local_features = local_features.tolist()

            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {
                    "train_loss": float(train_loss), 
                    "train_accuracy": float(train_accuracy),
                    "local_features": json.dumps(local_features),
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
    def __init__(self, node_id, net, trainloader, 
                 valloader, local_epochs, attack_type):
        super().__init__(node_id, net, trainloader, valloader, local_epochs)
        self.attack_type = attack_type

    def _generate_malicious_features(self, features: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 0.1, features.shape)
        return features + noise

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
    
        if self.attack_type != "Same_Value":
            
            self.net.train()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.net.parameters())

            running_loss = 0.0
            correct = 0
            total = 0

            for batch in self.trainloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.net(images)
                loss = criterion(outputs, labels)

                # For Gradient Ascent Attack
                if self.attack_type == "Gradient_Ascent":
                    loss = -loss  # Maximize loss instead of minimizing

                loss.backward()
                optimizer.step()
                running_loss += loss.item()


                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = float(running_loss / len(self.trainloader))
            train_accuracy = float(correct / total) if total > 0 else 0.0
        else:
       
            train_loss = 1.0
            train_accuracy = 0.0

        # Get current model parameters
        model_weights = get_weights(self.net)

        # For Same Value Attack
        if self.attack_type == "Same_Value":
            for i in range(len(model_weights)):
                model_weights[i] = np.ones_like(model_weights[i])

        # For Sign Flipping Attack
        elif self.attack_type == "Sign_Flipping":
            for i in range(len(model_weights)):
                model_weights[i] = -model_weights[i]

        features = compute_features(self.net, self.trainloader, self.device)
        malicious_features = self._generate_malicious_features(features)        
        # Return manipulated weights and metrics
        return (
            model_weights,
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "local_features": json.dumps(malicious_features.tolist()),
                "node_id": self.node_id
            }
        )

def client_fn(context: Context):
    # Load from config
    config = load_config()
    local_epochs = config['local-epochs']
    node_id = context.node_config['partition-id']

    train_loader, val_loader = load_data(
        partition_id=node_id, 
        num_partitions=context.node_config['num-partitions'],
        dataset=config['dataset']
    )
    net = MnistNet(config) if config['dataset'] == 'mnist' else  CifarNet(config)

    malicious_partition_ids = config['malicious_clients_id']

    if node_id in malicious_partition_ids:  
        return MaliciousClient(
            net=net,
            node_id = node_id,
            trainloader=train_loader,
            valloader=val_loader,
            attack_type=config['attack_type'],
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