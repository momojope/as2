import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIGSIZE = (6, 4)
LINEWIDTH = 2.0
FONTSIZE = 12
def plot_loss_accs(
    statistics, multiple_runs=False, log_x=False, log_y=False, 
    figsize=FIGSIZE, linewidth=LINEWIDTH, fontsize=FONTSIZE,
    fileName=None, filePath=None, show=True
    ):

    rows, cols = 1, 2
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))
    color_1 = 'tab:blue' # #1f77b4
    color_2 = 'tab:red' # #d62728
    
    same_steps = False
    if multiple_runs :
        all_steps = statistics["all_steps"]
        same_steps = all(len(steps) == len(all_steps[0]) for steps in all_steps) # Check if all runs have the same number of steps
        if same_steps :
            all_steps = np.array(all_steps[0]) + 1e-0 # Add 1e-0 to avoid log(0)
        else :
            all_steps = [np.array(steps) + 1e-0 for steps in all_steps] # Add 1e-0 to avoid log(0)
            color_indices = np.linspace(0, 1, len(all_steps))
            colors = plt.cm.viridis(color_indices)
    else :
        all_steps = np.array(statistics["all_steps"]) + 1e-0

    for i, key in enumerate(["accuracy", "loss"]) :
        ax = fig.add_subplot(rows, cols, i+1)
        if multiple_runs :
            zs = np.array(statistics["train"][key])
            if same_steps :
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                #ax.errorbar(all_steps, zs_mean, yerr=zs_std, fmt=f'-', color=color_1, label=f"Train", lw=linewidth)
                ax.plot(all_steps, zs_mean, '-', color=color_1, label=f"Train", lw=linewidth)
                ax.fill_between(all_steps, zs_mean-zs_std, zs_mean+zs_std, color=color_1, alpha=0.2)
            else :  
                for j, z in enumerate(zs) :
                    ax.plot(all_steps[j], z, '-', color=colors[j], label=f"Train", lw=linewidth/2)

            zs = np.array(statistics["test"][key])
            if same_steps :
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                ax.plot(all_steps, zs_mean, '-', color=color_2, label=f"Eval", lw=linewidth)
                ax.fill_between(all_steps, zs_mean-zs_std, zs_mean+zs_std, color=color_2, alpha=0.2)
            else :  
                for j, z in enumerate(zs) :
                    ax.plot(all_steps[j], z, '--', color=colors[j], label=f"Eval", lw=linewidth/2)

        else :
            ax.plot(all_steps, statistics["train"][key], "-", color=color_1,  label=f"Train", lw=linewidth) 
            ax.plot(all_steps, statistics["test"][key], "-", color=color_2,  label=f"Eval", lw=linewidth) 

        if log_x : ax.set_xscale('log')
        #if log_y : ax.set_yscale('log')
        if log_y and key=="loss" : ax.set_yscale('log') # No need to log accuracy
        ax.tick_params(axis='y', labelsize='x-large')
        ax.tick_params(axis='x', labelsize='x-large')
        ax.set_xlabel("Training Steps (t)", fontsize=fontsize)
        if key=="accuracy": s = "Accuracy"
        if key=="loss": s = "Loss"
        #ax.set_ylabel(s, fontsize=fontsize)
        ax.set_title(s, fontsize=fontsize)
        ax.grid(True)
        if multiple_runs and (not same_steps) :
            legend_elements = [Line2D([0], [0], color='k', lw=linewidth, linestyle='-', label='Train'),
                            Line2D([0], [0], color='k', lw=linewidth, linestyle='--', label='Eval')]
            ax.legend(handles=legend_elements, fontsize=fontsize)
        else :
            ax.legend(fontsize=fontsize)

    if fileName is not None and filePath is not None :
        os.makedirs(filePath, exist_ok=True)
        plt.savefig(f"{filePath}/{fileName}"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    if show : plt.show()
    else : plt.close()


def plot_scaling_results(results, r_train_values, model_types=['lstm', 'gpt'], seeds=[0, 42]):
    # Create plots directory
    plots_dir = '../plots/scaling_experiment'
    os.makedirs(plots_dir, exist_ok=True)

    metrics = {
        'train_loss': {'title': 'Training Loss', 'log_scale': True},
        'val_loss': {'title': 'Validation Loss', 'log_scale': True},
        'train_acc': {'title': 'Training Accuracy', 'log_scale': False},
        'val_acc': {'title': 'Validation Accuracy', 'log_scale': False},
        'train_loss_steps': {'title': 'Steps to Minimum Training Loss', 'log_scale': False},
        'val_loss_steps': {'title': 'Steps to Minimum Validation Loss', 'log_scale': False},
        'train_acc_steps': {'title': 'Steps to Maximum Training Accuracy', 'log_scale': False},
        'val_acc_steps': {'title': 'Steps to Maximum Validation Accuracy', 'log_scale': False}
    }

    # Set up colors and markers for each model
    colors = {'lstm': 'blue', 'gpt': 'red'}
    markers = {'lstm': 'o', 'gpt': 's'}

    # For each metric, create a plot showing both models
    for metric_key, metric_info in metrics.items():
        plt.figure(figsize=(10, 6))

        for model_type in model_types:
            # Extract data for each seed
            data_by_seed = []
            for seed in seeds:
                if metric_key == 'train_loss':
                    values = [results[model_type][seed][r]['extrema']['min_train_loss'] for r in r_train_values]
                elif metric_key == 'val_loss':
                    values = [results[model_type][seed][r]['extrema']['min_test_loss'] for r in r_train_values]
                elif metric_key == 'train_acc':
                    values = [results[model_type][seed][r]['extrema']['max_train_accuracy'] for r in r_train_values]
                elif metric_key == 'val_acc':
                    values = [results[model_type][seed][r]['extrema']['max_test_accuracy'] for r in r_train_values]
                elif metric_key == 'train_loss_steps':
                    values = [results[model_type][seed][r]['extrema']['min_train_loss_step'] for r in r_train_values]
                elif metric_key == 'val_loss_steps':
                    values = [results[model_type][seed][r]['extrema']['min_test_loss_step'] for r in r_train_values]
                elif metric_key == 'train_acc_steps':
                    values = [results[model_type][seed][r]['extrema']['max_train_accuracy_step'] for r in r_train_values]
                elif metric_key == 'val_acc_steps':
                    values = [results[model_type][seed][r]['extrema']['max_test_accuracy_step'] for r in r_train_values]

                data_by_seed.append(values)

            # Calculate mean and std
            mean_values = np.mean(data_by_seed, axis=0)
            std_values = np.std(data_by_seed, axis=0)

            # Plot with error bars
            plt.errorbar(r_train_values, mean_values, yerr=std_values,
            label=model_type.upper(), color=colors[model_type],
            marker=markers[model_type], linestyle='-', linewidth=2,
            markersize=8, capsize=5)

        plt.xlabel('Training Data Fraction (r_train)')
        plt.ylabel(metric_info['title'])
        plt.title(f'{metric_info["title"]} vs Training Data Fraction')
        plt.grid(True, alpha=0.3)
        plt.legend()

        if metric_info['log_scale']:
            plt.yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric_key}_vs_rtrain.png'), dpi=300)
        plt.close()

    # Create the combined plot for val_acc (with generalization threshold)
    plt.figure(figsize=(10, 6))
    for model_type in model_types:
        # Extract validation accuracy data for each seed
        data_by_seed = []
        for seed in seeds:
            values = [results[model_type][seed][r]['extrema']['max_test_accuracy'] for r in r_train_values]
            data_by_seed.append(values)

        # Calculate mean and std
        mean_values = np.mean(data_by_seed, axis=0)
        std_values = np.std(data_by_seed, axis=0)

        # Plot with error bars
        plt.errorbar(r_train_values, mean_values, yerr=std_values,
        label=model_type.upper(), color=colors[model_type],
        marker=markers[model_type], linestyle='-', linewidth=2,
        markersize=8, capsize=5)

    # Add generalization threshold line
    plt.axhline(y=0.9, color='green', linestyle='--', label='Generalization Threshold (0.9)')
    plt.axhline(y=0.5, color='orange', linestyle='--', label='Overfitting Threshold (0.5)')

    plt.xlabel('Training Data Fraction (r_train)')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Training Data Fraction')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'generalization_analysis.png'), dpi=300)
    plt.close()


# Function to analyze generalization thresholds
def analyze_generalization(results, r_train_values, model_types=['lstm', 'gpt'], seeds=[0, 42]):
    generalization_analysis = {}

    for model_type in model_types:
        generalization_analysis[model_type] = {}

        # Collect data across seeds
        all_val_accs = []
        all_train_accs = []

        for seed in seeds:
            val_accs = [results[model_type][seed][r]['extrema']['max_test_accuracy'] for r in r_train_values]
            train_accs = [results[model_type][seed][r]['extrema']['max_train_accuracy'] for r in r_train_values]
            all_val_accs.append(val_accs)
            all_train_accs.append(train_accs)

        # Average across seeds
        mean_val_accs = np.mean(all_val_accs, axis=0)
        mean_train_accs = np.mean(all_train_accs, axis=0)

        # Find generalization threshold (smallest r_train with val_acc >= 0.9)
        generalization_indices = np.where(mean_val_accs >= 0.9)[0]
        if len(generalization_indices) > 0:
            min_gen_idx = generalization_indices[0]
            generalization_threshold = r_train_values[min_gen_idx]
        else:
            generalization_threshold = "Not reached"

        # Find overfitting threshold (largest r_train with train_acc ≈ 1.0 and val_acc <= 0.5)
        overfitting_indices = np.where((mean_train_accs > 0.99) & (mean_val_accs <= 0.5))[0]
        if len(overfitting_indices) > 0:
            max_overfit_idx = overfitting_indices[-1]
            overfitting_threshold = r_train_values[max_overfit_idx]
        else:
            overfitting_threshold = "Not observed"

        generalization_analysis[model_type] = {
            'generalization_threshold': generalization_threshold,
            'overfitting_threshold': overfitting_threshold
        }

    return generalization_analysis