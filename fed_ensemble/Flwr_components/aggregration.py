import json
import torch
import numpy as np
from fed_ensemble.task import load_config
from flwr.server.strategy import FedAvg, FedProx
from fed_ensemble.DGM.ensemble_model import EnsembleModel
from fed_ensemble.AdaptiveThreshold import AdaptiveThreshold
from fed_ensemble.Utils.FilesMetricsManager import file_metrics_manager


class OverrideFedAvg(FedAvg if load_config()['strategy'] == 'FedAVG' else FedProx):
    def __init__(
        self,
        fraction_fit=0.5,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=None,
        ensembleModel: EnsembleModel = None,
        evaluate_fn: callable = None, 
        config = None, 
    ):
        
        if config['strategy'] == 'FedAVG':
            super().__init__(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_available_clients=min_available_clients,
                initial_parameters=initial_parameters,
                evaluate_fn=evaluate_fn
            )
        else:
            super().__init__(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_available_clients=min_available_clients,
                initial_parameters=initial_parameters,
                evaluate_fn=evaluate_fn,
                proximal_mu=0.01  # Default proximal_mu if not provided
            )
        self.current_weights = initial_parameters
        self.ensembleModel = ensembleModel
        self.baseline_established = False
        self.thresholder = AdaptiveThreshold()
        self.client_reputation = {}
        self.config = config

    def update_repuation(self, node_id, anomaly_score, threshold):

        self.client_reputation[node_id] = (self.client_reputation[node_id] * 0.98 
                                        if anomaly_score > threshold 
                                        else min(1.0, self.client_reputation[node_id] + 0.01))
            

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and update current weights."""
        
        if not results:
            return None, {}
        
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
                
        # Initiating baseline training 
        if not self.baseline_established:
            node_updates = np.concatenate([update['features'] for index, update in enumerate(client_updates) 
                                            ],axis=0)
            
            self.ensembleModel.train_model(
                node_updates,
                baseline_established= self.baseline_established,
                epochs=25
            ) 
            self.baseline_established = True
        
        # Compute the anoamly scores obtain for each node
        anomaly_scores = []
        metrics = file_metrics_manager.get_metrics()
        for idx, update in enumerate(client_updates):
            features_tensor = torch.tensor(update['features'], dtype=torch.float32, 
                                                device=self.ensembleModel.device)
           
            node_id = update['node_id']
            if node_id not in self.client_reputation:
                self.client_reputation[node_id] = 1.0
        
            scores = self.ensembleModel.compute_anomaly(
                features_tensor,
                repuation=self.client_reputation[node_id]
            )

            client_idx = next((i for i, client in enumerate(metrics['clients']) 
                      if client['node_id'] == node_id), -1)
    
            current_score = float(np.mean(scores.detach().cpu().numpy()))
            current_loss = float(update['metrics'].metrics['train_loss'])
            current_accuracy = float(update['metrics'].metrics['train_accuracy'])
            
            if client_idx == -1:
                # Add new client
                metrics['clients'].append({
                    'node_id': node_id,
                    'anomaly_scores': current_score,
                    'loss': [current_loss],
                    'accuracy': [current_accuracy]
                })
            else:
                # Update existing client
                metrics['clients'][client_idx]['anomaly_scores'] = current_score
                metrics['clients'][client_idx]['loss'].append(current_loss)
                metrics['clients'][client_idx]['accuracy'].append(current_accuracy)
            
            anomaly_scores.append({
                "node_id": update['node_id'],
                "scores": scores
            }) 

        file_metrics_manager.update_client_metrics(metrics)
        current_loss = metrics.get("training_progress", [{}])[-1].get("loss", 0)

        flatten_scores = []
        for client in anomaly_scores:
            print(f"Client {client['node_id']} score {np.mean(client['scores'].detach().cpu().numpy()):.3f}")
            flatten_scores.extend(client['scores'].detach().cpu().numpy().flatten())

        threshold = self.thresholder.compute_threshold(
            anomaly_scores=flatten_scores,
            loss_performed=current_loss
        )

        print(f"\nThreshold: {threshold:.3f}\n")       

        filtered_results = []
        for result, client in zip(results, anomaly_scores):
            score = np.mean(client['scores'].detach().cpu().numpy())
            self.update_repuation(client['node_id'], score, threshold)
            if (score < threshold):
                filtered_results.append(result)

        print(f"\nFiltered results: {len(filtered_results)}\n")

        valid_features = np.concatenate([client['features'] 
                                         for client in extraction(filtered_results)], 
                                         axis=0)

        if len(filtered_results):
            self.ensembleModel.train_model(
                features=valid_features,
                baseline_established=True,
                epochs=20
            )
            # Re-aggregrate the results 
            aggregate_weights = super().aggregate_fit(
                server_round=server_round, 
                results=filtered_results,
                failures=failures
            )
            self.current_weights = aggregate_weights[0]
            return aggregate_weights[0], {}
        
        return None, {}
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results from clients."""
        if not results:
            return None, {}
        
        # Aggregate metrics weighted by the number of examples
        precisions = []
        recalls = []
        f1_scores = []
        losses = []
        examples = []

        for client_id, client_result in results:
            if "precision" in client_result.metrics and "loss" in client_result.metrics:
                precisions.append(client_result.metrics["precision"] * client_result.num_examples)
                recalls.append(client_result.metrics["recall"] * client_result.num_examples)
                f1_scores.append(client_result.metrics["f1"] * client_result.num_examples)
                losses.append(client_result.metrics["loss"] * client_result.num_examples)
                examples.append(client_result.num_examples)

        # Compute weighted averages
        precision = sum(precisions) / sum(examples)
        recall = sum(recalls) / sum(examples)
        f1 = sum(f1_scores) / sum(examples)
        loss = sum(losses) / sum(examples)

        return round(loss, 3), {
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3),
        }
    