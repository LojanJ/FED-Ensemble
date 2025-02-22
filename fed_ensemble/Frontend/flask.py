from flask import Flask
from flask_cors import CORS
from server_app import metrics_data

app = Flask(__name__)
CORS(app)

@app.route('/metrics')
def get_metrics():
    return metrics_data

if __name__ == '__main__':
    app.run(port=8000)