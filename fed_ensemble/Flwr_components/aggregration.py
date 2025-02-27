import json
import torch
import numpy as np
from scipy import stats
from flwr.server.strategy import FedAvg
from fed_ensemble.DGM.ensemble_model import EnsembleModel
from fed_ensemble.adaptiveThreshold import AdaptiveThreshold

class OverrideFedAvg(FedAvg):
    def __init__(
        self,
        fraction_fit=0.5,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=None,
        ensembleModel: EnsembleModel = None,
        evaluate_fn: callable = None, 
        config = None,
        trusted_node = 0,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn
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
            self.ensembleModel.train_model(
                trusted_node[0]['features'],
                epochs=25
            )
            self.baseline_established = True
        
        # Compute the anoamly scores obtain for each node
        anomaly_scores = []
        for idx, update in enumerate(client_updates):
            features_tensor = torch.tensor(update['features'], dtype=torch.float32, 
                                                device=self.ensembleModel.device)
           
            scores = self.ensembleModel.compute_anomaly_score(
                features_tensor
            )
            
            anomaly_scores.append({
                "node_id": update['node_id'],
                "scores": scores
            })
        
        # print('ANOMALY_SCORES', anomaly_scores)
        anomaly_flag = self.thresholder.anomaly_flag(
            anomaly_scores=anomaly_scores
        )

        print('ANOMALY_FLAG', anomaly_flag)
        for client in anomaly_scores:
            print(f"Client {client['node_id']} score {np.mean(client['scores'].detach().cpu().numpy()):.3f}")
        

        filtered_results = [ result for result, client in 
                                zip(results, anomaly_scores) 
                                if np.mean(client['scores'].detach().cpu().numpy()) < 
                                anomaly_flag['threshold']]


        valid_features = np.concatenate([client['features'] 
                                         for client in extraction(filtered_results)], 
                                         axis=0)

        if valid_features.size & server_round % 2 == 0:
            self.ensembleModel.train_model(
                features=valid_features,
                epochs=25
            )
        # Re-aggregrate the results 
        aggregate_weights = super().aggregate_fit(
            server_round=server_round, 
            results=filtered_results,
            failures=failures
        )
        self.current_weights = aggregate_weights

        return aggregate_weights if 'aggregrate_weights' in locals() else baseline_weights, {}
   

    def aggregate_evaluate(self, server_round, results, failures):
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
    

