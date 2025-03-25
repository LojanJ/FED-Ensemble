import json
import os
import threading
import time
from pathlib import Path
import numpy as np

class FileMetricsManager:
    def __init__(self):
        current_dir = Path(__file__).parent
        self.file_path = current_dir / 'metrics.json'
        self.file_lock = threading.Lock()
        
        # Initialize with empty metrics if file doesn't exist
        if not os.path.exists(self.file_path):
            self._write_metrics({
                "clients": [],
                "training_progress": [],
                "performance": {
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "timestamp": ""
                }
            })

    def _read_metrics(self):
        """Read metrics from file with lock protection"""
        with self.file_lock:
            try:
                with open(self.file_path, 'r') as f:
                    metrics = json.load(f)
                    return metrics
            except (json.JSONDecodeError, FileNotFoundError):
                # Return default metrics if file is corrupted or doesn't exist
                return {
                    "clients": [],
                    "training_progress": [],
                    "performance": {
                        "cpu_percent": 0.0,
                        "memory_percent": 0.0,
                        "timestamp": ""
                    }
                }
    
    def _write_metrics(self, metrics):
        """Write metrics to file with lock protection"""
        with self.file_lock:
            with open(self.file_path, 'w') as f:
                json.dump(self._convert_numpy_types(metrics), f, indent=2)
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def update_performance(self, cpu, memory, timestamp):
        """Update performance metrics"""
        metrics = self._read_metrics()
        metrics["performance"].update({
            "cpu_percent": cpu,
            "memory_percent": memory,
            "timestamp": timestamp
        })
        
        self._write_metrics(metrics)
    
    def update_client_metrics(self, clients_data):
        """Update client metrics"""
        metrics = self._read_metrics()
        metrics["clients"] = clients_data.get("clients", [])
        self._write_metrics(metrics)
    
    def update_server_metrics(self, server_data):
        """Update server metrics"""
        metrics = self._read_metrics()
        metrics["training_progress"] = server_data.get("training_progress", [])
        self._write_metrics(metrics)
    
    def get_metrics(self):
        """Get current metrics""" 
        return self._read_metrics()
    
    def rest_metrics(self):
        self._write_metrics({
            "clients": [],
            "training_progress": [],
            "performance": {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "timestamp": ""
            }
        })

# Create a singleton instance
file_metrics_manager = FileMetricsManager()