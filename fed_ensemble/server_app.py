from flask import Flask, jsonify
from fed_ensemble.aggregration import OverrideFedAvg
from fed_ensemble.ensemble_model import EnsembleModel
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from fed_ensemble.task import Net, get_weights, load_config
import threading
import time

# Rename Flask app to avoid conflict
flask_app = Flask(__name__)

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

@flask_app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics_data)

def start_flask_app():
    flask_app.run(port=8000)

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
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        noise_dim=config['noise_dim'],
        device=config['device'],
        lr=config['lr'],
        gamma=config['gamma']
    )

    # Use custom strategy
    strategy = OverrideFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        ensembleModel=ensemble_model,
        config=config
    )

    return ServerAppComponents(strategy=strategy, config=server_config)

def main():
    # Launch Flask app in a separate thread
    flask_thread = threading.Thread(target=start_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Give the Flask app a moment to start
    time.sleep(1)

    # Create and start ServerApp
    server_app = ServerApp(server_fn=server_fn)
    server_app.start()

def update_metrics(new_metrics):
    """Update metrics data with new values"""
    global metrics_data
    metrics_data.update(new_metrics)

if __name__ == "__main__":
    main()