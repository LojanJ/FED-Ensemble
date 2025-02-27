import numpy as np
import torch

class AdaptiveThreshold:

    def __init__(self,  
                 min_threshold=0.1, 
                 max_threshold=0.9):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.historical_scores = []
        self.window_size = 5

    def anomaly_flag(self, anomaly_scores):
        scores = [client['scores'] for client in anomaly_scores]
        threshold = self.compute_threshold_stastical(scores)

        client_metrics = []
        for client in anomaly_scores:
            scores_np = client['scores'].detach().cpu().numpy()
            client_metrics.append({
                'node_id': client['node_id'],
                'mean_score': float(np.mean(scores_np)),
                'max_score': float(np.max(scores_np)),
                'score_std': float(np.std(scores_np))
            })

        return {
            'threshold': float(threshold),
            'client_metrics': client_metrics   
        }

    def compute_threshold_stastical(self, scores):
        flat_scores = []
        for score_tensor in scores:
            if torch.is_tensor(score_tensor):
                flat_scores.extend(score_tensor.detach().cpu().numpy().flatten())
            else:
                flat_scores.extend(score_tensor.flatten())
        
        scores_arr = np.array(flat_scores)

        # Calculate basic statistics
        q1, q3 = np.percentile(scores_arr, [25, 75])
        iqr = q3 - q1

        print('\nHistory threhold', self.historical_scores, '\nIQR, Q3, Q1:', iqr, q3, q1)
        # Adaptive threshold based on distribution
        if len(self.historical_scores) >= self.window_size:
            historical_std = np.std(self.historical_scores)
            base_threshold = q3 + 1.5 * iqr * (1 + historical_std)
        else:
            base_threshold = q3 + 1.5 * iqr
    
        # Keep threshold history
        self.historical_scores.append(np.mean(scores_arr))
        if len(self.historical_scores) > self.window_size:
            self.historical_scores.pop(0)
            
        print('\nFinal threhosld',base_threshold)
        return np.clip(base_threshold, self.min_threshold, self.max_threshold)


 