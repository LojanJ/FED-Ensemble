import threading
from flask import Flask, jsonify, request
from flask_cors import CORS
import psutil
import sys
import os
from datetime import datetime
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Now import project modules
from fed_ensemble.Utils.FilesMetricsManager import file_metrics_manager
from fed_ensemble.task import load_config

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000"   # React default port
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})  # Enable CORS for React frontend
process = psutil.Process()

def update_performance_metrics():
    """Periodically update CPU and memory usage."""
    while True:
        try:
            cpu_percent = process.cpu_percent(interval=5.0)
            print(cpu_percent)
            memory_info = process.memory_info()
            memory_percent = (memory_info.rss / psutil.virtual_memory().total) * 100

            file_metrics_manager.update_performance(
                cpu_percent,
                memory_percent,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            time.sleep(5)
        except Exception as e:
            print(f"Error updating metrics: {e}")
            time.sleep(5)

# Start performance monitoring in background
threading.Thread(target=update_performance_metrics, daemon=True).start()

@app.route('/metrics', methods=['GET'])
def get_metrics():
    metrics = file_metrics_manager.get_metrics()
    return jsonify(metrics)

@app.route('/config', methods=['GET'])
def get_config():
    try:
        config = load_config()
        return jsonify(config), 200
    except Exception as e:
        print(f"Error loading config: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

