import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results(attack_type, dataset, strategy='FedAVG'):
    base_path = f"../results/{dataset}/wo_FED-Ensemble"
    results = {}

    files = [f for f in os.listdir(base_path) if f"{dataset}_{strategy}_{attack_type}_20" in f]
    for file_name in files:
        p_num = file_name.split('_p')[-1]
        with open(os.path.join(base_path, file_name), 'r') as f:
            results[f'p{p_num}'] = json.load(f)
    
    return results

def plot_metrics(attack_type, dataset, metric='loss', strategy='FedAVG'):
    results = load_results(attack_type, dataset, strategy)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    colors = [
        '#4CAF50',  
        '#F44336', 
        '#2196F3',  
        '#FFC107',  
        '#9C27B0'  
    ]
    
    for (p_num, data), color in zip(results.items(), colors):
        rounds = [r['round'] for r in data['training_progress']]
        values = [r[metric] for r in data['training_progress']]
        if metric == 'loss':
            values = np.clip(values, 1e-8, None)
        
        p_num = str(p_num).split('p')[1]
        plt.plot(rounds, values, color=color, label=f'No of Poisoned Clients: {p_num}', 
                marker='o', markersize=4, linewidth=2)
    
    plt.title(f'{dataset} {strategy} {attack_type}: {metric.capitalize()} Comparison', 
             color='white', pad=20)
    plt.xlabel('Training Rounds', color='white')
    plt.ylabel(metric.capitalize(), color='white')
    plt.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
    
    legend = plt.legend(frameon=True)
    legend.get_frame().set_facecolor('#2F2F2F')
    legend.get_frame().set_edgecolor('white')
    
    ax.tick_params(colors='white')
    
    save_dir = f"../results/plots/{dataset}/{attack_type}"
    save_path = f"{save_dir}/{metric}_comparison.png"
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Remove existing file if it exists
    if os.path.exists(save_path):
        os.remove(save_path)
    
    plt.savefig(save_path, 
                dpi=300, 
                bbox_inches='tight',
                facecolor='black',
                edgecolor='none')
    plt.close()
    
    plt.style.use('default')


def generate_all_plots():
    datasets = ['MNIST', 'CIFAR']
    metrics = ['loss', 'accuracy']
    attack_types = ['GradAscent', 'SameValue', 'SignFlip', 'NoAttck']
    
    # Original individual plots
    for dataset in datasets:
        for attack in attack_types:
            for metric in metrics:
                plot_metrics(attack, dataset, metric)

if __name__ == "__main__":
    generate_all_plots()