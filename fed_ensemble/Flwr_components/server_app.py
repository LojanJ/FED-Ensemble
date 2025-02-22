from flask import Flask, jsonify
from fed_ensemble.Flwr_components.aggregration import OverrideFedAvg
from fed_ensemble.DGM.ensemble_model import EnsembleModel
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from datasets import load_dataset
from fed_ensemble.task import (
    Net, 
    get_weights, 
    set_weights, 
    load_config, test, 
    apply_transforms)
from torch.utils.data import DataLoader
import torch

# Rename Flask app to avoid conflict
# flask_app = Flask(__name__)

# Global variable to store metrics
metrics_data = {
    "num_clients": 0,
    "num_rounds": 0,
    "global_accuracy": 0.0,
    "global_loss": 0.0,
    "malicious_clients_ratio": 0.0,
    "clients": [],
    "training_progress": []  # Add this to store training progress
}

# @flask_app.route('/metrics', methods=['GET'])
# def get_metrics():
#     return jsonify(metrics_data)

# def start_flask_app():
#     flask_app.run(port=8000)

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        config = load_config('fed_ensemble/config.json') 
        net = Net(config)
        set_weights(net, parameters_ndarrays)
        net.to(device)
        evaluate = test(net, testloader, device=device)
        return evaluate['loss'], {"centralized_accuracy": evaluate['accuracy'],
                      "centralized_precision": evaluate['precision'],
                      "centralized_recall": evaluate['recall'],
                      "centralized_f1": evaluate['f1_score']}

    return evaluate

def server_fn(context: Context) -> ServerAppComponents:   
    # Load config
    config = load_config('fed_ensemble/config.json')
    num_rounds = config['num-server-rounds']
    fraction_fit = config['fraction-fit']

    # Initialize model
    net = Net(config)
    parameters = ndarrays_to_parameters(get_weights(net))

    # Server configuration
    server_config = ServerConfig(num_rounds=num_rounds)
    ensemble_model = EnsembleModel(
        lr = config['lr'],
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        noise_dim=config['noise_dim'],
        n_models=2,
        device=config['device']
    )

    global_test_set = load_dataset('mnist')['test']

    test_loader = DataLoader(
        global_test_set.with_transform(apply_transforms),
        batch_size=32
    )
    # Use custom strategy
    strategy = OverrideFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_fn= gen_evaluate_fn(testloader=test_loader, device=config['device']),
        ensembleModel=ensemble_model,
        config=config
    )

    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(server_fn=server_fn)