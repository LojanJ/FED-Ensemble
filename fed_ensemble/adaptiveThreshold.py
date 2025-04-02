# In AdaptiveThreshold.py
import numpy as np
import torch


class AdaptiveThreshold:
    historical_scores=[]
    def __init__(
        self,
        initial_percentile=55,
        min_percentile=50,
        max_percentile=70,
        min_threshold=0.0,
        max_threshold=1.0
    ):
        self.current_percentile = initial_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.previous_performance = None  
 
    def compute_threshold(self, anomaly_scores, loss_performed):
        # Add exponential moving average smoothing
        if len(self.historical_scores) > 0:
            smoothed_scores = 0.8 * np.array(self.historical_scores[-5:]).mean() + 0.2 * np.array(anomaly_scores).mean()
            anomaly_scores = np.concatenate([anomaly_scores, [smoothed_scores]])

        loss_improvement = self.previous_performance - loss_performed if self.previous_performance else 0
        if loss_improvement > 0.01: 
            self.current_percentile = min(
                self.max_percentile,
                self.current_percentile + 2
            )
        elif loss_improvement < -0.01:
            self.current_percentile = max(
                self.min_percentile,
                self.current_percentile - 3
            )

        # Compute base threshold with small buffer for handling few poisoned clients
        threshold = np.percentile(anomaly_scores, self.current_percentile)
        
        self.previous_performance = loss_performed
        # Maintain historical thershold
        self.historical_scores.append(threshold)
        # Ensure threshold stays within bounds
        return threshold