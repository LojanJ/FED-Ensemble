from flask import Flask, jsonify
from flask_cors import CORS
import random
import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your React frontend

@app.route('/metrics', methods=['GET'])
def get_metrics():
    metrics = {
        "num_clients": 10,
        "num_rounds": 10,
        "global_accuracy": round(random.uniform(0.5, 0.9), 2),
        "global_loss": round(random.uniform(0.1, 0.5), 2),
        "poison_type": "Gradient Ascent",
        "clients": [
            {"id": i, "accuracy": round(random.uniform(0.4, 0.9), 2), 
             "anomaly_score": round(random.uniform(0.0, 1.0), 2), 
             "loss": round(random.uniform(0.2, 0.7), 2)}
            for i in range(1, 11)
        ],
        "training_progress": [
            {"rounds": i, "accuracy": round(random.uniform(0.3, 0.9), 2), 
             "loss": round(random.uniform(0.05, 0.5), 2)}
            for i in range(10)
        ], 
        "last_updated": datetime.datetime.now().isoformat()
    }
    return jsonify(metrics)

@app.route('/configuration', methods=['GET'])
def get_config():
    config = {
        "num-server-rounds": 10,
        "fraction-fit": 0.5,
        "malicious_clients_id": [3, 7],
        "lr": 0.001,
        "input_dim": 784,
        "latent_dim": 128,
        "noise_dim": 64,
        "device": "cpu",
        "batch_size": 32
    }

    return jsonify(config)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)