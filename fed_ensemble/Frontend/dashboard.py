import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from task import load_config

# Config parameters
CONFIG = load_config(config_path='./config.json')

# Page configuration
st.set_page_config(
    page_title="Federated Learning Monitor",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🌐 FED-Ensemble: Federated Learning System Monitor")
st.markdown("Real-time monitoring of federated learning metrics and model performance")

# Simplified mock data generation
def fetch_metrics():
    try:
        response = requests.get('http://localhost:8000/metrics')
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)
        metrics = response.json()
    except requests.exceptions.RequestException as e:
        # st.error(f"Error fetching metrics from server: {e}") # Use st.error
        metrics = { # Fallback data
            "num_clients": 10,
            "num_rounds": CONFIG["num-server-rounds"],
            "global_accuracy": .7,
            "global_loss": .2,
            "malicious_clients_ratio": len(CONFIG['malicious_clients_id'])/10,
            "clients": [
                {"accuracy": 0.6, "anomaly_score": 0.1, "loss": 0.5},
                {"accuracy": 0.7, "anomaly_score": 0.20, "loss": 0.5},
                {"accuracy": 0.4, "anomaly_score": 0.30, "loss": 0.6}
            ],
            "training_progress": [
                {"rounds": 0, "accuracy": 0.35, "loss": 0.10},
                {"rounds": 1, "accuracy": 0.70, "loss": 0.04},
                {"rounds": 2, "accuracy": 0.50, "loss": 0.04},
                {"rounds": 3, "accuracy": 0.40, "loss": 0.04},
                {"rounds": 4, "accuracy": 0.70, "loss": 0.04},
                {"rounds": 5, "accuracy": 0.70, "loss": 0.04},
                {"rounds": 6, "accuracy": 0.70, "loss": 0.04},
                {"rounds": 7, "accuracy": 0.70, "loss": 0.04},
                {"rounds": 8, "accuracy": 0.70, "loss": 0.04},
                {"rounds": 9, "accuracy": 0.70, "loss": 0.04}
            ]
        }
    return metrics
# Fetch metrics
metrics = fetch_metrics()

# Top metrics row
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1%}</div>
            <div class="metric-label">Global Accuracy</div>
        </div>
    """.format(metrics["global_accuracy"]), unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1%}</div>
            <div class="metric-label">Global Loss</div>
        </div>
    """.format(metrics["global_loss"]), unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1%}</div>
            <div class="metric-label">Malicious Client Ratio</div>
        </div>
    """.format(metrics["malicious_clients_ratio"]), unsafe_allow_html=True)

# Training Progress Section
st.header("📈 Training Progress")
progress_df = pd.DataFrame(metrics["training_progress"])

progress_fig = go.Figure()
progress_fig.add_trace(go.Scatter(
    x=progress_df["rounds"],
    y=progress_df["accuracy"],
    name="Accuracy",
    line=dict(color="#1f77b4", width=2)
))
progress_fig.add_trace(go.Scatter(
    x=progress_df["rounds"],
    y=progress_df["loss"],
    name="Loss",
    line=dict(color="#ff7f0e", width=2)
))

progress_fig.update_layout(
    title="Global Model Performance Over Time",
    xaxis_title="Round",
    yaxis_title="Value",
    hovermode="x unified",
    height=400
)
st.plotly_chart(progress_fig, use_container_width=True)

# Client Performance Section
st.header("👥 Client Performance")
tab1, tab2 = st.tabs(["Performance Table", "Anomaly Distribution"])

with tab1:
    client_df = pd.DataFrame(metrics["clients"])
    
    def color_anomaly_score(val):
        color = 'red' if val > 0.7 else 'orange' if val > 0.4 else 'green'
        return f'color: {color}'
    
    styled_df = client_df.style\
        .format({
            'accuracy': '{:.1%}',
            'loss': '{:.3f}',
            'anomaly_score': '{:.2f}'
        })\
        .applymap(color_anomaly_score, subset=['anomaly_score'])
    
    st.dataframe(styled_df, use_container_width=True)

with tab2:
    fig_anomaly = px.histogram(
        client_df,
        x="anomaly_score",
        nbins=20,
        title="Distribution of Client Anomaly Scores"
    )
    fig_anomaly.update_layout(
        xaxis_title="Anomaly Score",
        yaxis_title="Number of Clients",
        height=400
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)

# Security Monitoring Section
st.header("🔒 Security Monitoring")
col1, col2 = st.columns(2)

with col1:
    threshold_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=metrics["malicious_clients_ratio"],
        title={'text': "Current Malicious Client Ratio"},
        gauge={'axis': {'range': [0, 1]},
               'steps': [
                   {'range': [0, 0.3], 'color': "lightgreen"},
                   {'range': [0.3, 0.7], 'color': "orange"},
                   {'range': [0.7, 1], 'color': "red"}
               ],
               'threshold': {
                   'line': {'color': "red", 'width': 4},
                   'thickness': 0.75,
                   'value': 0.7
               }}))
    threshold_fig.update_layout(height=300)
    st.plotly_chart(threshold_fig, use_container_width=True)

with col2:
    anomaly_levels = ['Low', 'Medium', 'High']
    anomaly_counts = [
        sum(1 for score in client_df['anomaly_score'] if score <= 0.4),
        sum(1 for score in client_df['anomaly_score'] if 0.4 < score <= 0.7),
        sum(1 for score in client_df['anomaly_score'] if score > 0.7)
    ]
    anomaly_fig = px.pie(
        values=anomaly_counts,
        names=anomaly_levels,
        title="Anomaly Level Distribution"
    )
    anomaly_fig.update_layout(height=300)
    st.plotly_chart(anomaly_fig, use_container_width=True)

# System Health Section
st.header("🔧 System Health")
col1, col2 = st.columns(2)

with col1:
    cpu_usage = np.random.uniform(20, 80)
    st.progress(cpu_usage/100, text=f"CPU Usage: {cpu_usage:.1f}%")

with col2:
    memory_usage = np.random.uniform(30, 90)
    st.progress(memory_usage/100, text=f"Memory Usage: {memory_usage:.1f}%")

# Model Configuration Display
st.header("⚙️ Model Configuration")
config_df = pd.DataFrame([CONFIG]).T.reset_index()
config_df.columns = ['Parameter', 'Value']
st.dataframe(config_df, use_container_width=True)

# Footer with last update time
st.markdown("---")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")