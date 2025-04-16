import numpy as np
from fed_ensemble.flwr_components.aggregration import OverrideFedAvg
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from datasets import load_dataset
from fed_ensemble.task import (
    CifarNet,
    MnistNet,
    get_weights, 
    set_weights, 
    load_config, test, 
    apply_transforms)
from torch.utils.data import DataLoader
import torch
from fed_ensemble.Utils.FilesMetricsManager import file_metrics_manager
from fed_ensemble.AdaptiveThreshold import AdaptiveThreshold
from fed_ensemble.DGM.ensemble_model import EnsembleModel

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    threshold: AdaptiveThreshold,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        config = load_config() 
        net = MnistNet(config) if config['dataset'] == 'mnist' else  CifarNet(config)
        set_weights(net, parameters_ndarrays)
        net.to(device)
        evaluate = test(net, testloader, device=device)

        metrics = file_metrics_manager.get_metrics()
        training_progress = metrics.get("training_progress", [])

        new_entry = {
            "round": server_round,
            "accuracy": float(evaluate["accuracy"]), 
            "loss": float(evaluate["loss"]),
            "precision": float(evaluate["precision"]),
            "recall": float(evaluate["recall"]),
            "f1": float(evaluate["f1_score"]),
            "threshold": threshold.historical_scores[-1] if threshold.historical_scores else 0
        }

        if training_progress:
            median_metrics = {
                "median_accuracy": np.median([entry.get("accuracy", 0.0) for entry in training_progress]),
                "median_loss": np.median([entry.get("loss", 0.0) for entry in training_progress]),
                "median_precision": np.median([entry.get("precision", 0.0) for entry in training_progress]),
                "median_recall": np.median([entry.get("recall", 0.0) for entry in training_progress]),
                "median_f1": np.median([entry.get("f1", 0.0) for entry in training_progress])
            }
            
            print("\nMedian Metrics Summary:")
            print(f"Median Accuracy: {median_metrics['median_accuracy']:.4f}")
            print(f"Median Loss: {median_metrics['median_loss']:.4f}")
            print(f"Median Precision: {median_metrics['median_precision']:.4f}")
            print(f"Median Recall: {median_metrics['median_recall']:.4f}")
            print(f"Median F1 Score: {median_metrics['median_f1']:.4f}\n")

        # Update existing entry or append new one
        for entry in training_progress:
            if entry["round"] == server_round:
                entry.update(new_entry)
                break
        else:
            training_progress.append(new_entry)
            
        # Update metrics with the possibly modified training_progress
        file_metrics_manager.update_server_metrics({"training_progress": training_progress})

        return evaluate['loss'], {"centralized_accuracy": evaluate['accuracy'],
                      "centralized_precision": evaluate['precision'],
                      "centralized_recall": evaluate['recall'],
                      "centralized_f1": evaluate['f1_score']}

    return evaluate

def server_fn(context: Context) -> ServerAppComponents:   

    file_metrics_manager.rest_metrics()
    
    # Load config
    config = load_config()
    num_rounds = config['num-server-rounds']
    fraction_fit = config['fraction-fit']

    # Global threshold instance
    threshold = AdaptiveThreshold(
        min_percentile=config['min_percentile'],
        max_percentile=config['max_percentile'],
        initial_percentile=config['initial_percentile']
    )

    # Initialize model
    net = MnistNet(config) if config['dataset'] == 'mnist' else  CifarNet(config)
    parameters = ndarrays_to_parameters(get_weights(net))

    # Server configuration
    server_config = ServerConfig(num_rounds=num_rounds)
    ensemble_model = EnsembleModel(
        lr = config['ensemble_lr'],
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        noise_dim=config['noise_dim'],
        n_models=3,
        device=config['device']
    )

    if torch.cuda.device_count() > 1:
        torch.nn.DataParallel(ensemble_model)

    global_test_set = load_dataset('mnist' if config['dataset'] == 'mnist' else 'cifar10')['test']
    print(f"Number of centralized testing {len(global_test_set)}")
    test_loader = DataLoader(
        global_test_set.with_transform(lambda x: apply_transforms(x, config['dataset'])),
        batch_size=32
    )

    # Use custom strategy
    strategy = OverrideFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_fn= gen_evaluate_fn(testloader=test_loader, device=config['device'], threshold=threshold),
        ensembleModel=ensemble_model,
        config=config
    )

    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(server_fn=server_fn)