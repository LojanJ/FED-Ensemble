# In AdaptiveThreshold.py
import numpy as np
import torch

class AdaptiveThreshold:
    def __init__(
        self,
        initial_percentile=60,
        min_percentile=50,
        max_percentile=65,
        min_threshold=0.0,
        max_threshold=1.0
    ):
        self.current_percentile = initial_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.historical_scores = [] 
        self.previous_performance = None  
 
    def compute_threshold(self, anomaly_scores, loss_performed):
        # Compute base threshold
        threshold = np.percentile(anomaly_scores, self.current_percentile)
        
        # Adjust percentile based on loss_performed change
        if loss_performed is not None and self.previous_performance is not None:
            print(loss_performed, self.previous_performance, loss_performed < self.previous_performance, self.current_percentile)
            if loss_performed > self.previous_performance:
                # Decrease percentile (stricter) if performance drops
                self.current_percentile = max(
                    self.min_percentile, 
                    self.current_percentile - 5
                )
            else:
                # Increase percentile (more lenient) if performance improves
                self.current_percentile = min(
                    self.max_percentile, 
                    self.current_percentile + 1
                )
        
        self.previous_performance = loss_performed
        # Maintain historical thershold
        self.historical_scores.append(threshold)
        # Ensure threshold stays within bounds
        return threshold