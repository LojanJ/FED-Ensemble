# Fed-Ensemble: A Flower / PyTorch app

## Abstract

Federated Learning (FL) endorses promising avenues for organizations, enabling collaborative model
training across distributive network of clients, exempting client relaying raw data across the grid of
network, hence preserving users' privacy and lowering communication overheads. Albeit the proactive
benefits were evident, the decentralized nature of FL systems turns out highly susceptible from adversarial
threats, particularly Poisoning Attack (PA) raised by the networks' edge. Preventive countermeasures
against PA are often crafted to align with the specific client behaviors, consequentially drafting a tailored
response. Thereby, the integrity of global model stands at a risk, with ever-evolving interest of malefactors
exploiting the networks' vulnerabilities.

The author attempts to bridge this gap, proposing a feature representation learner utilizing
a novel unsupervised detective framework, leveraging Ensemble of Deep Generative Models (EDGMs)
to deduce suspicious scores on collected model updates, eliminating irregularities and aggregating
untampered or legitimate model updates. The foremost objective of the proposed framework entails,
deflecting anomalies via continuous representation learning and develop a feasible solution derived from
the manifested theoretical models.

The system assessed through brief testing, examined under minimal experimental configurations, aligned
with authors' resource availability. Identifying satisfiable outcomes, achieving detection accuracy varying
90% under MNIST dataset and 70-80% under CIFAR-10 dataset, with conservative minimization of
ubiquitous clients infected by 20-40% rationale of poisoning. Future enhancements explore client data
diversification, improve efficiency and scaling system feasibility.

**Keywords:** Federated Learning security, Poisoning Attacks, Unsupervised Learning, Ensemble Learning

**Subject Descriptors:**
- Behavioral Outlier Methodologies → Normality Feature Learning → Ensemble Deep Generative Learning Models
- FL privacy and security → Client Selection Strategy → Eliminate Adversarial Threats

## Project Structure

```
fed_ensemble/
├── adaptiveThreshold.py   # Adaptive thresholding implementation
├── client/                # Federated client components
├── config.json            # Main configuration file
├── config/                # Configuration directory
├── data/                  # Data storage
├── DGM/                   # Deep Generative Models implementation
├── Flwr_components/       # Flower framework components
├── Frontend/              # Dashboard and visualization components
│   ├── server.py          # Flask server for the dashboard
│   └── fed-ensemble-dashboard/ # React-based dashboard
├── models/                # ML model implementations
├── results/               # Result storage
├── server/                # Federated server components
├── task.py                # Main task implementation
└── utils/                 # Utility functions and metrics
```

## Install dependencies and project

```bash
pip install -e .
```

## Run with the Simulation Engine

In the `Fed-Ensemble` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
