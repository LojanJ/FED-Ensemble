import json
import torch
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
from fed_ensemble.DGM.ensemble_model import EnsembleModel
from fed_ensemble.adaptiveThreshold import AdaptiveThreshold
from fed_ensemble.utils.FilesMetricsManager import file_metrics_manager

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
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients, 
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn
        )
        self.thresholder = AdaptiveThreshold(
            initial_percentile=config['initial_percentile'],
            min_percentile=config['min_percentile'],
            max_percentile=config['max_percentile'],
        )
        self.current_weights = initial_parameters
        self.ensembleModel = ensembleModel
        self.baseline_established = False
        self.client_reputation = {}
        self.DACC = {
            'accuracy': [],
            'tpr': [],
            'fpr': [],
            'fnr': []
        }
        self.config = config
        
        # Performance and weights history
        self.history = [] 
        self.history_size = 5  
        self.min_improvement = 0.01  
        self.max_drop = 0.2 
        self.rollback_count = 0

    def update_reputation(self, node_id, anomaly_score, threshold):
        self.client_reputation[node_id] = (max(0.8, self.client_reputation[node_id] * 0.98)
                                           if anomaly_score > threshold 
                                           else min(1.0, self.client_reputation[node_id] + 0.01))

    def check_for_rollback(self, server_round, current_loss, current_accuracy):
        self.history.append((server_round, self.current_weights, current_loss, current_accuracy))
    
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
    
        if len(self.history) < 3:
            return False
        
        accuracies = [acc for _, _, _, acc in self.history]
        best_accuracy = max(accuracies)
        current_accuracy = accuracies[-1]
        avg_accuracy = np.mean(accuracies)
        
        # Calculate improvement or drop relative to the best in the window
        improvement = current_accuracy - min(accuracies[:-1])  
        relative_drop = (best_accuracy - current_accuracy) / (best_accuracy + 1e-8)  # Avoid division by zero
        
        # Conditions for rollback
        stagnation = improvement < self.min_improvement and current_accuracy < best_accuracy
        significant_drop = relative_drop > self.max_drop
        
        if significant_drop or stagnation:
            print(f"Performance issue detected at Round {server_round}:")
            if significant_drop:
                print(f"  Significant drop: {relative_drop:.3f} > {self.max_drop} "
                      f"(best: {best_accuracy:.3f}, current: {current_accuracy:.3f})")
            if stagnation:
                print(f"  Stagnation: improvement {improvement:.3f} < {self.min_improvement}, "
                      f"current {current_accuracy:.3f} < best {best_accuracy:.3f}")
            print(f"Average accuracy in window: {avg_accuracy:.3f}")
            self.rollback_to_best()
            return True
        
        return False

    def rollback_to_best(self):
        if not self.history:
            return None
        
        best_round, best_weights, best_loss, best_accuracy = max(self.history, key=lambda x: x[3])
        self.current_weights = best_weights
        self.rollback_count += 1
        print(f"Rolled back to weights from Round {best_round} with accuracy {best_accuracy:.3f} "
              f"(Rollback #{self.rollback_count}). Ensemble model and reputation scores preserved.")

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        
        aggregate_weights = super().aggregate_fit(
                server_round=server_round, 
                results=results,
                failures=failures
            )
        
        def extraction(client_result):
            client_updates = []
            for idx, (client_result, client_metrics) in enumerate(results):
                if "local_features" in client_metrics.metrics.keys():
                    try:
                        features = np.array(json.loads(client_metrics.metrics["local_features"]))
                        if features.size > 0:
                            features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
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

        anomaly_scores = []
        metrics = file_metrics_manager.get_metrics()
        for idx, update in enumerate(client_updates):
            features_tensor = torch.tensor(update['features'], dtype=torch.float32, 
                                           device=self.ensembleModel.device)
            node_id = update['node_id']
            if node_id not in self.client_reputation:
                self.client_reputation[node_id] = 1.0

            scores, adjusted_scores = self.ensembleModel.compute_anomaly(
                features_tensor,
                reputation=self.client_reputation[node_id]
            )

            client_idx = next((i for i, client in enumerate(metrics['clients']) 
                              if client['node_id'] == node_id), -1)
            current_score = float(np.mean(adjusted_scores.detach().cpu().numpy()))
            current_loss = float(update['metrics'].metrics['train_loss'])
            current_accuracy = float(update['metrics'].metrics['train_accuracy'])
            
            
            if client_idx == -1:
                metrics['clients'].append({
                    'node_id': node_id,
                    'anomaly_scores': [current_score],
                    'loss': [current_loss],
                    'accuracy': [current_accuracy]
                })
            else:
                metrics['clients'][client_idx]['anomaly_scores'].append(current_score)
                metrics['clients'][client_idx]['loss'].append(current_loss)
                metrics['clients'][client_idx]['accuracy'].append(current_accuracy)
            
            anomaly_scores.append({
                "node_id": update['node_id'],
                "scores": scores,
                'reputation_scores': adjusted_scores
            }) 

        file_metrics_manager.update_client_metrics(metrics)
        current_loss = metrics.get("training_progress", [{}])[-1].get("loss", 0)
        
        flatten_scores = []
        for client in anomaly_scores:
            raw_score = np.mean(client['scores'].detach().cpu().numpy())
            adjusted_score = np.mean(client['reputation_scores'].detach().cpu().numpy())
            print(f"Client {client['node_id']} Raw score {raw_score:.3f} Adjusted score {adjusted_score:.3f}")
            flatten_scores.extend(client['scores'].detach().cpu().numpy().flatten())
            

        threshold = self.thresholder.compute_threshold(
            anomaly_scores=flatten_scores,
            loss_performed=current_loss
        )
        print(f"\nThreshold: {threshold:.3f}",
        f"Percentile: {self.thresholder.current_percentile}\n")      

        # Primary filtering with threshold
        filtered_results = []
        for i, (result, client) in enumerate(zip(results, anomaly_scores)):
            score = np.mean(client['reputation_scores'].detach().cpu().numpy())
            self.update_reputation(client['node_id'], score, threshold)
            
            if score < threshold:
                filtered_results.append(result)
                print(f"Client {client['node_id']} accepted with score {score:.3f}")

        if len(filtered_results) == 0:
            print("All clients rejected! Using all clients instead.")
            filtered_results = results

        print(f"\nFinal filtered results: {len(filtered_results)}/{len(results)}\n")
        DACC = self.detection_accuracy([client['node_id'] for client in client_updates], 
                               [result[1].metrics['node_id'] for result in filtered_results])
       
        metrics = file_metrics_manager.get_metrics()
        training_progress = metrics.get("training_progress", [])
        
        training_progress.append({
            "round": server_round,
            "detection_accuracy": DACC,
        })
            
        file_metrics_manager.update_server_metrics({"training_progress": training_progress})

        valid_features = np.concatenate([client['features'] 
                                        for client in extraction(filtered_results)], 
                                        axis=0)

        if len(filtered_results):
            self.ensembleModel.train_model(features=valid_features, epochs=15)
            aggregate_weights = super().aggregate_fit(
                server_round=server_round, 
                results=filtered_results,
                failures=failures
            )
            if aggregate_weights[0] is not None:
                self.current_weights = aggregate_weights[0]

            # Evaluate and check for rollback
            current_weights_ndarrays = parameters_to_ndarrays(self.current_weights)
            loss, metrics_dict = self.evaluate_fn(server_round, current_weights_ndarrays, {})
            if self.check_for_rollback(server_round, loss, metrics_dict["centralized_accuracy"]):
                return self.current_weights, {}
            
            return aggregate_weights[0], {}
        
        return None, {}
    
    def detection_accuracy(self, updates, filtered_nodes):
        tp, tn, fp, fn = 0, 0, 0, 0
        for id in updates:
            is_malicious = id in self.config['malicious_clients_id']
            is_filtered = id in filtered_nodes
            if is_malicious and not is_filtered:
                tp += 1  
            elif not is_malicious and not is_filtered:
                fp += 1  
            elif not is_malicious and is_filtered:
                tn += 1 
            elif is_malicious and is_filtered:
                fn += 1  
        print(tp, fp, tn, fn)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        self.DACC['accuracy'].append(accuracy)  
        self.DACC['tpr'].append(tpr)
        self.DACC['fpr'].append(fpr)
        self.DACC['fnr'].append(fnr)
        print(f"Accuracy={np.mean(self.DACC['accuracy']):.3f}, "
              f"TPR={np.mean(self.DACC['tpr']):.3f}, "
              f"FPR={np.mean(self.DACC['fpr']):.3f}, "
              f"FNR={np.mean(self.DACC['fnr']):.3f}")
        return accuracy

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        precisions, recalls, f1_scores, losses, examples = [], [], [], [], []
        for client_id, client_result in results:
            if "precision" in client_result.metrics and "loss" in client_result.metrics:
                precisions.append(client_result.metrics["precision"] * client_result.num_examples)
                recalls.append(client_result.metrics["recall"] * client_result.num_examples)
                f1_scores.append(client_result.metrics["f1"] * client_result.num_examples)
                losses.append(client_result.metrics["loss"] * client_result.num_examples)
                examples.append(client_result.num_examples)
        precision = sum(precisions) / sum(examples)
        recall = sum(recalls) / sum(examples)
        f1 = sum(f1_scores) / sum(examples)
        loss = sum(losses) / sum(examples)
        return round(loss, 3), {
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3),
        }