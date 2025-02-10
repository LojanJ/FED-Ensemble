import json
import torch
import numpy as np
from scipy import stats
from flwr.server.strategy import FedAvg
from fed_ensemble.ensemble_model import EnsembleModel

class OverrideFedAvg(FedAvg):
    def __init__(
        self,
        fraction_fit=0.5,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=None,
        ensembleModel: EnsembleModel = None,
        config = None,
        trusted_node = 0
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters, 
        )
        self.current_weights = initial_parameters
        self.ensembleModel = ensembleModel
        self.trusted_node = trusted_node
        self.baseline_established = False
        self.thresholder = AdaptiveThreshold()
        self.config = config

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and update current weights."""
        baseline_weights, _ = super().aggregate_fit(
            server_round, results, failures
        )
        
        if not results:
            return baseline_weights, {}
        
        def extraction(client_result):
            # Process feature collection and processing
            client_updates = []

            for idx, (client_result, client_metrics) in enumerate(results):
        
                if "local_features" in client_metrics.metrics.keys():
                    try:
                        features = np.array(json.loads(client_metrics.metrics["local_features"]))
                        if features.size > 0:
                            # Apply basic statistical normalization
                            features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
                            
                            # Check for invalid values
                            if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                                client_updates.append({
                                    "node_id": client_metrics.metrics.get("node_id", idx),
                                    "features": features,
                                    "result": client_result,
                                    "metrics": client_metrics
                                })

                    except (ValueError, KeyError) as e:
                        print(f"Error processing client {idx}: {e}")

            return client_updates
        
        client_updates = extraction(results)
        if not client_updates:
            return None, {}

        # Initiating baseline training with the trusted node
        if not self.baseline_established:
            trusted_node = [update for index, update in enumerate(client_updates) if index == self.trusted_node]
            self.ensembleModel.train_ensemble(
                trusted_node[0]['features'],
                epochs=50,
                noise_dim=self.config["noise_dim"]
            )
            self.baseline_established = True
        
        # Compute the anoamly scores obtain for each node
        anomaly_scores = []
        for idx, update in enumerate(client_updates):
            features_tensor = torch.tensor(update['features'], dtype=torch.float32, 
                                                device=self.ensembleModel.device)
            scores = self.ensembleModel.compute_anomaly_score(
                features_tensor,
                self.config["noise_dim"]
            )
            
            anomaly_scores.append({
                "node_id": update['node_id'],
                "scores": scores
            })
        print(anomaly_scores)
        anomaly_flag = self.thresholder.anomaly_flag(anomaly_scores=anomaly_scores)
        print(f"Malicious ratio: {anomaly_flag['malicious_ratio']}, Threshold: {anomaly_flag['threshold']}")
        for client in anomaly_scores:
            print(f"Client {client['node_id']} score {np.mean(client['scores'])}")
        

        if anomaly_flag['action'] == 0:
            filtered_results = results
        elif anomaly_flag['action'] == 1:
            filtered_results = [ result for result, client in zip(results, anomaly_scores) if np.mean(client['scores']) < anomaly_flag['threshold']]
        else:
            filtered_results = []

        valid_features = np.concatenate([client['features'] for client in extraction(filtered_results)], axis=0)
        if valid_features.size:
            self.ensembleModel.train_ensemble(
                all_features=valid_features,
                epochs=25,
                noise_dim=self.config["noise_dim"]

            )
        # Re-aggregrate the results 
        aggregate_weights = super().aggregate_fit(
            server_round=server_round, 
            results=filtered_results,
            failures=failures
        )
        self.current_weights = aggregate_weights

        return aggregate_weights if 'aggregrate_weights' in locals() else baseline_weights, {}

    def aggregate_evaluate(self, zserver_round, results, failures):
        """Aggregate evaluation results from clients."""
        if not results:
            return None, {}
        
        # Aggregate metrics weighted by the number of examples
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        losses = []
        examples = []

        for client_id, client_result in results:
            if "accuracy" in client_result.metrics and "loss" in client_result.metrics:
                accuracies.append(client_result.metrics["accuracy"] * client_result.num_examples)
                precisions.append(client_result.metrics["precision"] * client_result.num_examples)
                recalls.append(client_result.metrics["recall"] * client_result.num_examples)
                f1_scores.append(client_result.metrics["f1"] * client_result.num_examples)
                losses.append(client_result.metrics["loss"] * client_result.num_examples)
                examples.append(client_result.num_examples)

        if not accuracies:
            return None, {}

        # Compute weighted averages
        accuracy = sum(accuracies) / sum(examples)
        precision = sum(precisions) / sum(examples)
        recall = sum(recalls) / sum(examples)
        f1 = sum(f1_scores) / sum(examples)
        loss = sum(losses) / sum(examples)

        return round(loss, 3), {
            "Accuracy": round(accuracy, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3),
        }
    

class AdaptiveThreshold:

    def __init__(self, 
                 initial_malicious_ratio=0.1,  
                 min_threshold=0.1, 
                 max_threshold=0.9):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        self.historical_scores = []
        self.malicious_ratio = initial_malicious_ratio
        
        self.detection_sensitivity = 1.0
        self.distribution_entropy = 0.0 

    def _update_malicious_ratio(self, scores):
        """Update malicious ratio with more conservative estimation."""
        skewness = stats.skew(scores)
        kurtosis = stats.kurtosis(scores)
        
        hist, _ = np.histogram(scores, bins=10)
        probabilities = hist / np.sum(hist)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        print(f"Skewness: {skewness}, Kurtosis: {kurtosis}, Entropy: {entropy}")
        
        # More conservative ratio estimation
        estimated_ratio = (
            abs(skewness) * 0.2 + 
            entropy * 0.3 + 
            (kurtosis > 3) * 0.2
        ) * 0.5  # Scale down the overall estimation
        
        # Smoother update with stronger prior
        self.malicious_ratio = np.mean([
            0.8 * self.malicious_ratio,  # Stronger weight on previous estimate
            0.2 * np.clip(estimated_ratio, 0, 0.5)  # Cap at 50%
        ])
        
        return self.malicious_ratio

    def compute_threshold(self, anomaly_scores):
        """Compute threshold with improved adaptive logic."""
        scores = np.array(anomaly_scores)
        
        malicious_ratio = self._update_malicious_ratio(scores)
        
        if malicious_ratio < 0.1:  # Very low suspicious activity
            threshold = np.percentile(scores, 95)
        elif malicious_ratio < 0.3:  # Moderate suspicious activity
            threshold = np.percentile(scores, 85)
        else:  # High suspicious activity
            threshold = np.percentile(scores, 75)
        
        # Ensure reasonable bounds
        threshold = np.clip(threshold, 0.3, 0.8)
        
        return threshold
    
    def anomaly_flag(self, anomaly_scores):
        """
        Provide actionable recommendations based on the current state of anomaly scores
        """
        scores = [client.get('scores') for client in anomaly_scores]
        malicious_ratio = self._update_malicious_ratio(scores)
        threshold = self.compute_threshold(scores)
        
        return {
            'malicious_ratio': malicious_ratio,
            'threshold': threshold,
            'action': self._determine_action(malicious_ratio)
        }
    
    def _determine_action(self, malicious_ratio):
        """
        Suggest system actions based on the estimated proportion of malicious clien 
        """
        if malicious_ratio < 0.2:
            return 0
        elif 0.2 <= malicious_ratio < 0.5:
            return 1
        else:
            return 2