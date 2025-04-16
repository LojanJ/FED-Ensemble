
import numpy as np

class AdaptiveThreshold:
    historical_scores=[]
    def __init__(
        self,
        min_percentile,
        max_percentile,
        initial_percentile,
    ):
        self.current_percentile = initial_percentile
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.previous_performance = None  


    def compute_threshold(self, anomaly_scores, loss_performed):
        # Add exponential moving average smoothing
        if len(self.historical_scores) > 0:
            smoothed_scores = 0.8 * np.array(self.historical_scores[-5:]).mean() 
            + 0.2 * np.array(anomaly_scores).mean()
            anomaly_scores = np.concatenate([anomaly_scores, [smoothed_scores]])

        loss_improvement = (self.previous_performance - loss_performed 
                            if self.previous_performance else 0)
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

        threshold = np.percentile(anomaly_scores, self.current_percentile)
        self.previous_performance = loss_performed
        self.historical_scores.append(threshold)
        return threshold
    