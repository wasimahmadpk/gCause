import math
import os
import json
import random
import pathlib
import parameters
import re
import numpy as np
import seaborn as sns
import pandas as pd
from math import sqrt
from datetime import datetime
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from mvlearn.embed import MCCA
from scipy.special import stdtr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from matplotlib import pyplot as plt
import xarray as xr
#from tigramite import data_processing as pp


# Function to recursively convert numpy types to native Python types
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)  # Convert numpy int64 to regular Python int
    else:
        return obj
    


def calculate_mean_metrics(data):
    """
    Calculate the mean of each metric across all entries for each method.
    
    Parameters:
        data (dict): Dictionary containing the data for methods (e.g., PCMCI, VAR-GC).
    
    Returns:
        dict: A dictionary with mean values for each metric for each method.
    """
    mean_metrics = {}
    
    for method, metrics in data.items():
        # Initialize a dictionary to accumulate sums
        summed_metrics = {key: [] for key in metrics[next(iter(metrics))].keys()}
        
        # Collect values for each metric
        for record in metrics.values():
            for key, value in record.items():
                summed_metrics[key].append(value)
        
        # Calculate mean for each metric
        mean_metrics[method] = {key: np.mean(values) for key, values in summed_metrics.items()}
    
    return mean_metrics


def plot_boxplots(methods_metrics_dict, plot_path, filename="method_metrics.json"):

    # Ensure the plot path exists
    os.makedirs(plot_path, exist_ok=True)

    # Flatten the data for conversion to a DataFrame
    flattened_data = []

    for method, experiments in methods_metrics_dict.items():
        for experiment, metrics in experiments.items():
            for metric, value in metrics.items():
                flattened_data.append({"Method": method, "Experiment": experiment, "Metric": metric, "Value": value})

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(flattened_data)
    # print(f'Mean val: {df.mean()}')

    # Ensure the output directory exists
    os.makedirs(plot_path, exist_ok=True)

    # Convert methods_metrics_dict to ensure all numpy types are converted
    methods_metrics_dict = convert_numpy_types(methods_metrics_dict)

    # Save to JSON (preserve exact structure)
    json_full_path = os.path.join(plot_path, filename)
    with open(json_full_path, "w") as json_file:
        json.dump(methods_metrics_dict, json_file, indent=4)
    print(f"Metrics saved to JSON: {json_full_path}")
    # df.to_csv(csv_full_path, index=False)
    print(f'Mean metrics: {calculate_mean_metrics(methods_metrics_dict)}')

    # Step 3: Plot and save boxplots as PDF files for each metric
    metrics = df["Metric"].unique()
    for metric in metrics:
        # plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(figsize=(12, 6))
        # Filter DataFrame for current metric
        metric_data = df[df["Metric"] == metric]
        # Create a boxplot with Method on the x-axis and Value on the y-axis
        metric_data.boxplot(column="Value", by="Method", grid=False)
        # plt.title(f"Boxplot for {metric}")
        plt.suptitle("")  # Remove the automatic 'by' title
        plt.xlabel("Method", fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel(metric, fontsize=14)
        ax.set_yticks(np.arange(0, 1.10, 0.10))  # Set finer ticks
        ax.set_ylim(0, 1.09)
        plt.yticks(fontsize=14)

        # Construct full PDF path and save the plot
        pdf_filename = f"boxplot_{metric}.pdf"
        pdf_full_path = os.path.join(plot_path, pdf_filename)
        plt.savefig(pdf_full_path, format="pdf")
        plt.close()



def generate_causal_graph(n):
    # Create an n x n array with random 0s and 1s
    array = np.random.randint(0, 2, size=(n, n))
    
    # Set the diagonal elements to 1
    np.fill_diagonal(array, 1)

    # Remove upper triangle (set to 0)
    for i in range(n):
        for j in range(i + 1, n):
            array[i][j] = 0
            
    return array

def calculate_shd(ground_truth, predicted):
    """
    Calculates the Structural Hamming Distance (SHD) between two DAG adjacency matrices.
    SHD is the number of differing edges between the ground truth and predicted matrices.
    
    :param ground_truth: np.ndarray - Ground truth adjacency matrix (n x n).
    :param predicted: np.ndarray - Predicted adjacency matrix (n x n).
    :return: int - The SHD between the two matrices.
    """
    if ground_truth.shape != predicted.shape:
        raise ValueError("Both matrices must have the same shape")
    
    # Element-wise comparison (1 where they differ, 0 where they are the same)
    differences = ground_truth != predicted
    
    # Sum the number of differing edges (the number of 1s in the differences matrix)
    shd = np.sum(differences)
    
    # Calculate the maximum possible number of edges (n * (n - 1) for an n x n adjacency matrix without self-loops)
    n = ground_truth.shape[0]
    max_edges = n * (n - 1)
    
    # Calculate normalized SHD
    normalized_shd = shd / max_edges
    
    return shd


def evaluate(actual, predicted):

    y_true_flat = [item for sublist in actual for item in sublist]
    y_pred_flat = [item for sublist in predicted for item in sublist]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    # Extract TP, TN, FP, FN from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate (Sensitivity)
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Negative Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    # Create a dictionary to store metrics
    shd = calculate_shd(np.array(actual), np.array(predicted))
    metrics = {
        # 'TP': tp,
        # 'TN': tn,
        # 'FP': fp,
        # 'FN': fn,
        'TPR': tpr,
        'TNR': tnr,
        'FPR': fpr,
        'FNR': fnr,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Fscore': f1_score,
        'SHD': shd
    }
    
    return metrics


def remove_diagonal_and_flatten(matrix):
    """
    Removes the diagonal elements of a 2D matrix and returns the remaining
    elements as a flattened list.

    Parameters:
        matrix (np.ndarray): A 2D NumPy array.

    Returns:
        list: A list of elements excluding the diagonal.
    """
    # Ensure the input is a 2D NumPy array
    if not isinstance(matrix, np.ndarray) or len(matrix.shape) != 2:
        raise ValueError("Input must be a 2D NumPy array.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Mask the diagonal and flatten the result
    # mask = np.ones(matrix.shape, dtype=bool)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask].tolist()


def plot_precision_recall_curve(y_true, dist, corr, plot_path):
    """
    Function to plot the Precision-Recall curve and AUC comparison for two methods based on their prediction scores.
    
    Parameters:
    - y_true: Actual binary labels (ground truth).
    - dist: Predicted scores for Method 1 (e.g., Distance-based method).
    - corr: Predicted scores for Method 2 (e.g., Correlation-based method, lower means better for label 1).
    - plot_path: Path where the Precision-Recall plot will be saved.
    """
    
    # Flatten the lists to make them suitable for Precision-Recall calculation
    y_true = [item for sublist in y_true for item in sublist]
    dist_scores = [item for sublist in dist for item in sublist]  # Flatten distance-based scores
    corr_scores = [item for sublist in corr for item in sublist]  # Flatten correlation-based scores

    # Invert correlation scores if lower values indicate positive class (label 1)
    # corr_scores = [1 - score for score in corr_scores]  # Inverting scores for correlation method

    # Compute Precision-Recall curve for Method 1 (using dist_scores)
    precision1, recall1, _ = precision_recall_curve(y_true, dist_scores)
    pr_auc1 = auc(recall1, precision1)

    # Compute Precision-Recall curve for Method 2 (using inverted corr_scores)
    precision2, recall2, _ = precision_recall_curve(y_true, corr_scores)
    pr_auc2 = auc(recall2, precision2)

     # Plot random classifier line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

    # Plot Precision-Recall curves
    plt.figure(figsize=(8, 6))
    plt.plot(recall1, precision1, color='orange', lw=2, label=f'gCDMI-1tier (AUC = {pr_auc1:.2f})')
    plt.plot(recall2, precision2, color='green', lw=2, label=f'gCDMI-2tier (AUC = {pr_auc2:.2f})')

    # Labels and formatting
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc='lower left', fontsize=13)
    plt.grid(alpha=0.3)

    # Save plot as high-quality PDF
    rnd = np.random.randint(1, 9999)
    filename = plot_path + f"pr_comparison_{rnd}.pdf"
    plt.savefig(filename, format="pdf", dpi=400, bbox_inches="tight")

    # Show plot


def plot_roc_curve(y_true, dist, corr, plot_path):
    """
    Function to plot the ROC curve and AUC comparison for two methods based on their KLD scores.

    Parameters:
    - y_true: Actual binary labels.
    - dist: Distance scores (higher value = more likely positive class).
    - corr: Correlation scores (lower value = more likely positive class).
    - plot_path: Path where the ROC plot will be saved.
    """
    # Flatten the lists to make them suitable for ROC curve calculation
    y_true = [item for sublist in y_true for item in sublist]
    dist_scores = [item for sublist in dist for item in sublist]  # Distance-based scores
    corr_scores = [item for sublist in corr for item in sublist]  # Correlation-based scores
    
    precision, recall, thresholds = precision_recall_curve(y_true, dist_scores)
    # Invert correlation scores if lower values indicate positive class
    # corr_scores = [1 - score for score in corr_scores]  # Invert scores for correlation
    
    # Compute ROC curve and AUC for both methods
    fpr1, tpr1, _ = roc_curve(y_true, dist_scores)
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(y_true, corr_scores)
    roc_auc2 = auc(fpr2, tpr2)

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, color='orange', lw=2, label=f'gCDMI-1tier (AUC = {roc_auc1:.2f})')
    plt.plot(fpr2, tpr2, color='green', lw=2, label=f'gCDMI-2tier (AUC = {roc_auc2:.2f})')

    # Plot random classifier line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

    # Labels and formatting
    plt.xlabel('FPR', fontsize=13)
    plt.ylabel('TPR', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc='lower right', fontsize=13)
    plt.grid(alpha=0.3)

    rnd = random.randint(1, 9999)
    filename = pathlib.Path(plot_path + f"roc_{rnd}.pdf")
    # Save as high-quality PDF
    plt.savefig(filename, format="pdf", dpi=400, bbox_inches="tight")



def f1_max(labs,preds):

    # F1 MAX
    # print(f'lab: {labs}, pred:{preds}')
    precision, recall, thresholds = precision_recall_curve(labs, preds)
    f1_scores = 2 * recall * precision / (recall + precision)
    f1_thresh = thresholds[np.argmax(f1_scores)]
    f1_score = np.nanmax(f1_scores)
    return f1_score, f1_thresh


def plot_multitests_metrics(metrics_data, plot_path, metric_name):
    # Group sizes are fixed
    group_sizes = list(metrics_data.keys())
    # Prepare the plot
    plt.figure(figsize=(10, 6))
    
    # Loop through each test type
    for test, metrics in metrics_data[group_sizes[0]].items():  # Assuming data for the first group size
        y_values = []
        print(f'Test: {test}')
        # Loop through group sizes
        for group_size in group_sizes:
            # Ensure the metric is available for the current group size and test
            if test in metrics_data[group_size]:
                metric_value = metrics_data[group_size][test].get(metric_name, np.nan)
            else:
                metric_value = np.nan
            y_values.append(metric_value)
        print(y_values)
        
        # Plot each test with corresponding y-values for the current group size
        plt.plot(group_sizes, y_values, marker='o', linestyle='-', label=test)
    
    # Customize plot appearance
    plt.xticks(group_sizes, fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Interaction density', fontsize=13)
    plt.ylabel(f'{metric_name}', fontsize=13)
    plt.ylim(0, 1.1)
    
    # Add legend
    plt.legend(fontsize=13, loc='upper left', ncol=3)
    
    # # Save the plot if needed
    if plot_path:
        filename = pathlib.Path(plot_path) / 'tests_comparison.pdf'
        plt.savefig(filename, format="pdf", dpi=600, bbox_inches="tight")
    
    # Show the plot
    # plt.show()


def plot_multitests_boxplot(dictdata, plotpath, metricsname):
    """
    Generates and saves a boxplot for the specified metric across different tests.
    
    Parameters:
    - dictdata: A dictionary containing the data for various tests and metrics.
    - plotpath: The path where the plot should be saved.
    - metricsname: The name of the metric to plot (e.g., 'Fscore', 'Precision', 'Recall').
    """
    
    # Prepare data for plotting
    test_names = []
    metric_values = []

    # Iterate through the dictionary to extract metric values
    for key in dictdata:
        for test_name, metrics in dictdata[key].items():
            test_names.append(test_name)
            metric_values.append(metrics[metricsname])

    # Create a DataFrame
    df = pd.DataFrame({
        'Test': test_names,
        metricsname: metric_values
    })

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Create a boxplot
    sns.boxplot(x='Test', y=metricsname, data=df)

    # Customize the plot
    plt.xticks(rotation=0, ha='right', fontsize=13)  # Set x-tick label size
    plt.yticks(fontsize=13)  # Set y-tick label size
    plt.xlabel('Test', fontsize=13)  # Set x-axis label font size
    plt.ylabel(metricsname, fontsize=13)  # Set y-axis label font size
    # plt.title(f'Boxplot of {metricsname} for Different Tests', fontsize=15)
    plt.tight_layout()

     # # Save the plot if needed
    if plotpath:
        filename = pathlib.Path(plotpath) / 'bp_tests_comparison.pdf'
        plt.savefig(filename, format="pdf", dpi=600, bbox_inches="tight")

    # Show the plot
    # plt.show()

# Function to compute metrics for each predicted graph and find the best one
def evaluate_best_predicted_graph(actual, predicted_list):

    # Create a mask for off-diagonal elements (diagonal elements are set to 0)
    n = actual.shape[0]
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, 0)  # Set diagonal to 0 to ignore

    best_metrics = None
    best_f1_score = -1  # Initialize with a low F1 score
    best_tpr = -1       # Initialize TPR for tiebreakers
    best_fpr = float('inf')  # Initialize FPR with high value for tiebreakers
    
    # Flatten actual graph once, since it's common for all predictions
    y_true_flat = actual[mask].tolist()
    # y_true_flat = [item for sublist in actual for item in sublist]
    
    for predicted in predicted_list:
         # Flatten predicted graph
        y_pred_flat = predicted[mask].tolist()
        # y_pred_flat = [item for sublist in predicted for item in sublist]

        # Calculate confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Negative Rate
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        shd = calculate_shd(np.array(actual), np.array(predicted))

        # Check if this prediction has the highest F1 score, or if tied, check TPR and FPR
        if (f1_score > best_f1_score or
            (f1_score == best_f1_score and tpr > best_tpr) or
            (f1_score == best_f1_score and tpr == best_tpr and fpr < best_fpr)):
            
            # Update the best metrics
            best_f1_score = f1_score
            best_tpr = tpr
            best_fpr = fpr
            
            best_metrics = {
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TPR': tpr,
                'TNR': tnr,
                'FPR': fpr,
                'FNR': fnr,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Fscore': f1_score,
                'SHD': shd
            }
    
    return best_metrics


def test_link(actual, counterfactual, alpha=0.05):
    results = {}

    # 1. Kolmogorov-Smirnov (KS) Test
    ks_stat, ks_p = stats.ks_2samp(actual, counterfactual)
    results["KS Test"] = ks_p

    # 2. Mann-Whitney U Test
    mw_stat, mw_p = stats.mannwhitneyu(actual, counterfactual, alternative='two-sided')
    results["MWU Test"] = mw_p

    # 3. Student’s t-Test (Welch’s t-Test if variance is unequal)
    t_stat, t_p = stats.ttest_ind(actual, counterfactual, equal_var=False)
    results["t-Test"] = t_p

    # 4. Anderson-Darling Test
    ad_stat, ad_crit, ad_sig = stats.anderson_ksamp([actual, counterfactual])
    results["AD Test"] = ad_sig  # Returns the significance level

    # 5. Shapiro-Wilk Test (Normality Test for Actual Sample)
    shapiro_stat, shapiro_p = stats.shapiro(actual)
    results["SW Test"] = shapiro_p

    # 6. Cramér-von Mises Test
    cvm_result = stats.cramervonmises_2samp(actual, counterfactual)
    results["CM Test"] = cvm_result.pvalue

    # 7. Wilcoxon Signed-Rank Test (Non-parametric)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(actual, counterfactual)
    results["WSR Test"] = wilcoxon_p

    return results


def count_metrics(input_data):
    """
    Calculate TP, TN, FP, FN metrics for each method from the input data.

    Args:
        input_data (str or dict): Path to a JSON file or a dictionary containing metrics.

    Returns:
        dict: A dictionary containing aggregated TP, TN, FP, FN, and total experiments for each method.
    """
    # Check if the input is a file path or a dictionary
    if isinstance(input_data, str) and os.path.isfile(input_data):
        # Load the JSON data from the file
        with open(input_data, "r") as file:
            results = json.load(file)
    elif isinstance(input_data, dict):
        # Use the input as a dictionary directly
        results = input_data
    else:
        raise ValueError("Input must be a valid JSON file path or a dictionary.")

    # Initialize an empty dictionary to store the metrics for each method
    metrics_counts = {}

    # Iterate through the methods (e.g., "gCDMI", "var")
    for method, experiments in results.items():
        # Initialize counters for TP, TN, FP, FN
        tp_count = 0
        tn_count = 0
        fp_count = 0
        fn_count = 0

        # Iterate through each experiment and update the counters
        for exp in experiments.values():
            tp_count += exp.get("TP", 0)
            tn_count += exp.get("TN", 0)
            fp_count += exp.get("FP", 0)
            fn_count += exp.get("FN", 0)

        # Total number of experiments for the method
        total_experiments = len(experiments)

        # Store the metrics in a dictionary for the current method
        metrics_counts[method] = {
            "TP": tp_count,
            "TN": tn_count,
            "FP": fp_count,
            "FN": fn_count,
            "Total_Experiments": total_experiments
        }

    return metrics_counts



def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    else:
        return data




def plot_motor_metrics(data, pars, save_path='', json_path=''):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(json_path, exist_ok=True)

    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    data = convert_numpy_types(data)

    # Save raw metrics
    raw_path = os.path.join(save_path, f'{pars["motor_task"]}_metrics.json')
    with open(raw_path, 'w') as f:
        json.dump(data, f, indent=4)

    rows = []
    for movement, methods in data.items():
        for method, runs in methods.items():
            accuracies, fscores = [], []
            tprs, fprs, tnrs, fnrs = [], [], [], []

            for run_metrics in runs.values():
                accuracies.append(run_metrics["Accuracy"])
                fscores.append(run_metrics["Fscore"])
                tprs.append(run_metrics["TPR"])
                fprs.append(run_metrics["FPR"])
                tnrs.append(run_metrics["TNR"])
                fnrs.append(run_metrics["FNR"])

            rows.append({
                "Movement": movement,
                "Method": method,
                "Accuracy": np.mean(accuracies),
                "std_Accuracy": np.std(accuracies, ddof=1),
                "Fscore": np.mean(fscores),
                "std_Fscore": np.std(fscores, ddof=1),
                "TPR": np.mean(tprs),
                "std_TPR": np.std(tprs, ddof=1),
                "FPR": np.mean(fprs),
                "std_FPR": np.std(fprs, ddof=1),
                "TNR": np.mean(tnrs),
                "std_TNR": np.std(tnrs, ddof=1),
                "FNR": np.mean(fnrs),
                "std_FNR": np.std(fnrs, ddof=1)
            })

    df = pd.DataFrame(rows)

    # Save JSON with mean/std values
    mean_metrics_dict = {}
    for _, row in df.iterrows():
        movement = row["Movement"]
        method = row["Method"]
        if movement not in mean_metrics_dict:
            mean_metrics_dict[movement] = {}
        mean_metrics_dict[movement][method] = {
            "Accuracy": row["Accuracy"],
            "std_Accuracy": row["std_Accuracy"],
            "Fscore": row["Fscore"],
            "std_Fscore": row["std_Fscore"],
            "TPR": row["TPR"],
            "std_TPR": row["std_TPR"],
            "FPR": row["FPR"],
            "std_FPR": row["std_FPR"],
            "TNR": row["TNR"],
            "std_TNR": row["std_TNR"],
            "FNR": row["FNR"],
            "std_FNR": row["std_FNR"]
        }

    mean_path = os.path.join(json_path, f'mean_{pars["motor_task"]}_metrics.json')
    with open(mean_path, 'w') as f:
        json.dump(mean_metrics_dict, f, indent=4)

    # Plotting
    unique_methods = df["Method"].unique()
    method_colors = dict(zip(unique_methods, sns.color_palette("Set2", len(unique_methods))))

    def plot_metric(metric_name, ylabel):
        plt.figure(figsize=(12, 6))
        df_plot = df[["Movement", "Method", metric_name, f"std_{metric_name}"]].copy()
        df_plot.rename(columns={metric_name: "mean", f"std_{metric_name}": "std"}, inplace=True)
        df_plot["err"] = df_plot["std"] / 2

        ax = sns.barplot(
            data=df_plot,
            x="Movement", y="mean", hue="Method",
            palette=method_colors,
            ci=None
        )

        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            err = df_plot.iloc[i]["err"]
            if np.isnan(err) or err == 0:
                continue
            ymin = max(0, height - err)
            ymax = min(1, height + err)
            ax.errorbar(
                x=bar.get_x() + bar.get_width() / 2,
                y=height,
                yerr=[[height - ymin], [ymax - height]],
                fmt='none',
                c='black',
                capsize=5,
                lw=1.5
            )

        plt.xlabel("Task", fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax.set_yticks(np.arange(0, 1.10, 0.10))
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Method", loc='upper right', ncol=2)
        plt.tight_layout()

        pdf_path = os.path.join(save_path, f"mean_{metric_name.lower()}_barplot.pdf")
        plt.savefig(pdf_path, format='pdf')
        plt.close()

    plot_metric("Accuracy", "Accuracy")
    plot_metric("Fscore", "F1 Score")
    plot_metric("TPR", "True Positive Rate")
    plot_metric("FPR", "False Positive Rate")
    plot_metric("TNR", "True Negative Rate")
    plot_metric("FNR", "False Negative Rate")

def plot_motor_metrics2(data, pars, save_path='', json_path=''):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(json_path, exist_ok=True)

    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    data = convert_numpy_types(data)

    raw_path = os.path.join(save_path, 'motor_metrics.json')
    with open(raw_path, 'w') as f:
        json.dump(data, f, indent=4)

    rows = []
    for movement, methods in data.items():
        for method, metrics in methods.items():
            print(metrics)
            rows.append({
                "Movement": movement,
                "Method": method,
                "Accuracy": metrics["Accuracy"],
                "Fscore": metrics["Fscore"],
                "TPR": metrics["TPR"],
                "FPR": metrics["FPR"],
                "TNR": metrics["TNR"],
                "FNR": metrics["FNR"],
                "std_Accuracy": metrics.get("std_Accuracy", 0),
                "std_Fscore": metrics.get("std_Fscore", 0),
                "std_TPR": metrics.get("std_TPR", 0),
                "std_FPR": metrics.get("std_FPR", 0),
                "std_TNR": metrics.get("std_TNR", 0),
                "std_FNR": metrics.get("std_FNR", 0)
            })

    df = pd.DataFrame(rows)

    df_agg = df.groupby(['Movement', 'Method']).agg('mean').reset_index()

    mean_metrics_dict = {}
    for _, row in df_agg.iterrows():
        movement = row["Movement"]
        method = row["Method"]
        if movement not in mean_metrics_dict:
            mean_metrics_dict[movement] = {}
        mean_metrics_dict[movement][method] = {
            "Accuracy": row["Accuracy"],
            "std_Accuracy": row["std_Accuracy"],
            "Fscore": row["Fscore"],
            "std_Fscore": row["std_Fscore"],
            "TPR": row["TPR"],
            "std_TPR": row["std_TPR"],
            "FPR": row["FPR"],
            "std_FPR": row["std_FPR"],
            "TNR": row["TNR"],
            "std_TNR": row["std_TNR"],
            "FNR": row["FNR"],
            "std_FNR": row["std_FNR"]
        }

    mean_path = os.path.join(json_path, f'mean_{pars["motor_task"]}_metrics.json')
    print(mean_path)
    with open(mean_path, 'w') as f:
        json.dump(mean_metrics_dict, f, indent=4)

    unique_methods = df["Method"].unique()
    method_colors = dict(zip(unique_methods, sns.color_palette("Set2", len(unique_methods))))

    def plot_metric(metric_name, ylabel):
        plt.figure(figsize=(12, 6))
        df_plot = df_agg[["Movement", "Method", metric_name, f"std_{metric_name}"]].copy()
        df_plot.rename(columns={metric_name: "mean", f"std_{metric_name}": "std"}, inplace=True)
        df_plot["err"] = df_plot["std"] / 2

        ax = sns.barplot(
            data=df_plot,
            x="Movement", y="mean", hue="Method",
            palette=method_colors,
            ci=None
        )

        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            err = df_plot.iloc[i]["err"]
            if np.isnan(err) or err == 0:
                continue
            ymin = max(0, height - err)
            ymax = min(1, height + err)
            ax.errorbar(
                x=bar.get_x() + bar.get_width() / 2,
                y=height,
                yerr=[[height - ymin], [ymax - height]],
                fmt='none',
                c='black',
                capsize=5,
                lw=1.5
            )

        plt.xlabel("Task", fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.set_yticks(np.arange(0, 1.10, 0.10))
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Method", loc='upper right', ncol=len(unique_methods))
        plt.tight_layout()

        pdf_path = os.path.join(save_path, f"mean_{metric_name.lower()}_barplot.pdf")
        plt.savefig(pdf_path, format='pdf')
        plt.close()

    plot_metric("Accuracy", "Accuracy")
    plot_metric("Fscore", "F1 Score")
    plot_metric("TPR", "True Positive Rate")
    plot_metric("FPR", "False Positive Rate")
    plot_metric("TNR", "True Negative Rate")
    plot_metric("FNR", "False Negative Rate")



    
def convert_numpy_types(data):
    """
    Recursively convert numpy types to native Python types.
    """
    if isinstance(data, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        return float(data)
    elif isinstance(data, (np.ndarray,)):
        return data.tolist()  # Convert numpy arrays to lists
    else:
        return data


def average_repeated_movements(data):
    """
    Average the values for duplicate movement keys and for multiple experiments under each method.
    
    Parameters:
    - data (dict): Dictionary of metrics for movements and methods.

    Returns:
    - dict: Dictionary with averaged values for duplicate movements and experiments.
    """
    averaged_data = {}

    # Iterate through the data and group by movement key
    for movement, methods in data.items():
        if movement not in averaged_data:
            averaged_data[movement] = {}

        for method, experiments in methods.items():
            if method not in averaged_data[movement]:
                averaged_data[movement][method] = []

            # Collect all experiment metrics for the current method under the current movement
            for exp, metrics in experiments.items():
                # Check if this experiment's metrics exist in averaged_data, if not, add it
                averaged_data[movement][method].append(metrics)

    # Now, average the Accuracy and Fscore for each method under each movement
    for movement, methods in averaged_data.items():
        for method, metrics_list in methods.items():
            # Convert to numpy arrays to calculate the mean of each metric across experiments
            accuracy_values = np.array([metrics["Accuracy"] for metrics in metrics_list])
            fscore_values = np.array([metrics["Fscore"] for metrics in metrics_list])
            
            # Compute the mean of Accuracy and Fscore across all experiments
            avg_accuracy = np.mean(accuracy_values)
            avg_fscore = np.mean(fscore_values)

            # Store the averaged metrics for the movement-method pair
            averaged_data[movement][method] = {
                "Accuracy": avg_accuracy,
                "Fscore": avg_fscore
            }

    return averaged_data


# Function to convert timestamp to formatted date
def convert_timestamp(timestamp):
    # Assuming the timestamp format is 'YYMMDD'
    timestamp = str(timestamp)
    year = int(timestamp[:4])
    month = int(timestamp[4:6])
    day = int(timestamp[6:8])
    hour = int(timestamp[8:10])
    min = int(timestamp[10:12])

    # Convert to datetime object
    date_obj = datetime(year, month, day, hour, min)

    # Format the datetime object as 'DD-Mon-YYYY'
    formatted_date = date_obj.strftime('%d-%b-%Y %H:%M')

    return formatted_date

def get_shuffled_ts(SAMPLE_RATE, DURATION, root):
    # Number of samples in normalized_tone
    N = SAMPLE_RATE * DURATION
    yf = rfft(root)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)
    # plt.plot(xf, np.abs(yf))
    # plt.show()
    new_ts = irfft(shuffle(yf))
    return new_ts



def make_filename_safe(s):
    """
    Replace unsafe filename characters with underscores and clean up the string.
    """
    # Define characters not allowed in filenames (common unsafe chars)
    unsafe_chars = r'[<>:"/\\|?*\']'
    
    # Replace unsafe characters with underscore
    safe_str = re.sub(unsafe_chars, '_', s)
    
    # Replace spaces and multiple underscores with a single underscore
    safe_str = re.sub(r'\s+', '_', safe_str)        # spaces to underscore
    safe_str = re.sub(r'_+', '_', safe_str)         # multiple underscores to one
    
    # Strip underscores from start and end
    safe_str = safe_str.strip('_')
    
    return safe_str


def get_ground_truth(matrix, group_sizes):
    # Convert the input matrix to a numpy array
    matrix = np.array(matrix)
    
    # Determine the number of groups
    num_groups = len(group_sizes)
    
    # Initialize the reduced matrix with zeros
    reduced_matrix = np.zeros((num_groups, num_groups), dtype=int)
    
    # Determine the indices that belong to each group
    groups = []
    start_idx = 0
    for size in group_sizes:
        groups.append(list(range(start_idx, start_idx + size)))
        start_idx += size
    
    # Fill in the reduced matrix
    for i, group_i in enumerate(groups):
        for j, group_j in enumerate(groups):
            for var_i in group_i:
                for var_j in group_j:
                    if matrix[var_j, var_i] != 0:  # Check if var_j causes var_i
                        reduced_matrix[j, i] = 1  # Since rows are causes, update [j, i]
                        break  # No need to check further if one causal relationship is found
    
    # Ensure all groups have self-connection
    np.fill_diagonal(reduced_matrix, 1)
    return reduced_matrix


def deseasonalize(var, interval):
    deseasonalize_data = []
    for i in range(interval, len(var)):
        value = var[i] - var[i - interval]
        deseasonalize_data.append(value)
    return deseasonalize_data


def running_avg_effect(y, yint):

#  Break temporal dependency and generate a new time series
    pars = parameters.get_sig_params()
    SAMPLE_RATE = pars.get("sample_rate")  # Hertz
    DURATION = pars.get("duration")  # Seconds
    rae = 0
    for i in range(len(y)):
        ace = 1/((training_length + 1 + i) - training_length) * (rae + (y[i] - yint[i]))
    return rae


# Normalization (MixMax/ Standard)
def normalize(data, type='minmax'):

    if type == 'std':
        return (np.array(data) - np.mean(data))/np.std(data)

    elif type == 'minmax':
        return (np.array(data) - np.min(data))/(np.max(data) - np.min(data))


def down_sample(data, win_size, partition=None):
    agg_data = []
    daily_data = []
    for i in range(len(data)):
        daily_data.append(data[i])

        if (i % win_size) == 0:

            if partition == None:
                agg_data.append(sum(daily_data) / win_size)
                daily_data = []
            elif partition == 'gpp':
                agg_data.append(sum(daily_data[24: 30]) / 6)
                daily_data = []
            elif partition == 'reco':
                agg_data.append(sum(daily_data[40: 48]) / 8)
                daily_data = []
    return agg_data


def SNR(s, n):
    Ps = np.sqrt(np.mean(np.array(s) ** 2))
    Pn = np.sqrt(np.mean(np.array(n) ** 2))
    SNR = Ps / Pn
    return 10 * math.log(SNR, 10)


def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true))


def mutual_information(x, y):
    mi = mutual_info_regression(x, y)
    mi /= np.max(mi)
    return mi


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def read_ground_truth(file_path):
    """
    Reads a CSV file containing binary data with extra headers and index columns,
    discarding the first two rows and the first two columns, and returns a numpy array
    of the binary data.
    
    Parameters:
    - file_path (str): The path to the CSV file.
    
    Returns:
    - np.ndarray: A 2D numpy array containing the binary data.
    """
    # Read the CSV file, starting from the third row, and use the first column as the index
    df = pd.read_csv(file_path, header=1, index_col=0)
    
    # Drop the first column to get only the binary data
    binary_data_df = df.iloc[:, 1:]

    # Convert the DataFrame to a numpy array of integers
    binary_data = binary_data_df.to_numpy().astype(int)
    
    return binary_data.T


def plot_motor_count(data, save_path="plots", json_path=''):
    """
    Generate separate line plots and bar plots for each metric (TP, TN, FP, FN) for all methods.
    Each plot is saved as a separate PDF file, dynamically assigning markers and colors.
    Additionally, the plot data is saved as JSON files.

    Parameters:
    - data (dict): Dictionary containing metrics for movements and methods.
    - save_path (str): Directory path where the plots will be saved.
    """
    # Ensure save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Metrics to include in separate plots
    metrics = ["TP", "TN", "FP", "FN"]
    
    # List of markers for dynamic assignment (line plots)
    available_markers = ['o', 'v', 's', '^', 'D', 'P', '*', 'X', 'd', '<', '>']
    
    # Extract all unique methods dynamically
    example_movement = next(iter(data.keys()))  # Get the first movement key
    methods = list(data[example_movement].keys())  # Extract method names for that movement
    
    # Dynamically assign markers and colors to methods
    markers = {method: available_markers[i % len(available_markers)] for i, method in enumerate(methods)}
    method_colors = sns.color_palette("Set2", len(methods))  # Generate colors for the methods
    colors = {method: method_colors[i] for i, method in enumerate(methods)}
    
    # Iterate over each metric to create separate plots
    for metric in metrics:
        # Prepare data for line plot
        line_data = []
        for movement in data:
            for method in methods:
                line_data.append({
                    "Movement": movement,
                    "Method": method,
                    metric: data[movement][method][metric]
                })
        
        # Line Plot
        plt.figure(figsize=(12, 8))
        for method in methods:
            # Extract values for the current metric across movements
            values = [data[movement][method][metric] for movement in data]
            # Plot the values with a unique marker for each method
            plt.plot(
                data.keys(),
                values,
                marker=markers[method],
                color=colors[method],
                label=f"{method}",
                linewidth=2
            )
        
        # Add plot details
        plt.xlabel("Task", fontsize=14)
        plt.ylabel(f"{metric} Value", fontsize=14)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(bottom=0)  # Ensure the y-axis starts at 0
        plt.legend(title="Method", loc='upper right', fontsize=10, ncol=2)  # Adjust columns if too many methods
        plt.grid(True, linestyle='--', alpha=0.7)
        
        
        # Prepare data for bar plot
        bar_data = []
        for movement in data:
            for method in methods:
                bar_data.append({
                    "Movement": movement,
                    "Method": method,
                    metric: data[movement][method][metric]
                })
        bar_df = pd.DataFrame(bar_data)

        # Bar Plot
        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=bar_df,
            x="Movement",
            y=metric,
            hue="Method",
            palette=colors
        )
        
        plt.xlabel("Task", fontsize=14)
        plt.ylabel(f"{metric} Value", fontsize=14)
        plt.ylim(bottom=0)  # Ensure the y-axis starts at 0
        plt.xticks(rotation=0, ha='right', fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(title="Method", loc='upper right', fontsize=10, ncol=2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the bar plot as a PDF file
        bar_plot_file = os.path.join(save_path, f"{metric}_bar_plot_t.pdf")
        plt.tight_layout()
        plt.savefig(bar_plot_file, format='pdf')
        print(f"Saved {metric} bar plot at: {bar_plot_file}")
        plt.show()

        # Save bar data as JSON
        bar_data_j = convert_numpy_types(bar_data)
        bar_data_json = os.path.join(json_path, f"{metric}_bar_data.json")
        with open(bar_data_json, 'w') as json_file:
            json.dump(bar_data_j, json_file, indent=4)
        print(f"Saved {metric} bar data as JSON at: {bar_data_json}")


def kld(P, Q, epsilon=1e-10):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.

    Parameters:
    - P: np.ndarray, the first distribution
    - Q: np.ndarray, the second distribution
    - epsilon: float, small value added for numerical stability

    Returns:
    - kl_div: float, the Kullback-Leibler divergence from P to Q
    """
    # Step 1: Normalize the distributions
    P = P / np.sum(P) if np.sum(P) > 0 else P
    Q = Q / np.sum(Q) if np.sum(Q) > 0 else Q

    # Step 2: Add a small value to avoid log(0)
    P = P + epsilon
    Q = Q + epsilon

    # Step 3: Calculate KL Divergence
    kl_div = np.sum(P * np.log(P / Q))
    return round(kl_div, 3)

def calculate_shd(ground_truth, predicted):
    """
    Calculates the Structural Hamming Distance (SHD) between two DAG adjacency matrices.
    SHD is the number of differing edges between the ground truth and predicted matrices.
    
    :param ground_truth: np.ndarray - Ground truth adjacency matrix (n x n).
    :param predicted: np.ndarray - Predicted adjacency matrix (n x n).
    :return: int - The SHD between the two matrices.
    """
    if ground_truth.shape != predicted.shape:
        raise ValueError("Both matrices must have the same shape")
    
    # Element-wise comparison (1 where they differ, 0 where they are the same)
    differences = ground_truth != predicted
    
    # Sum the number of differing edges (the number of 1s in the differences matrix)
    shd = np.sum(differences)
    
    # Calculate the maximum possible number of edges (n * (n - 1) for an n x n adjacency matrix without self-loops)
    n = ground_truth.shape[0]
    max_edges = n * (n - 1)
    
    # Calculate normalized SHD
    normalized_shd = shd / max_edges
    return shd

def convert_variable_name(variable_name):
    # Use regular expression to find and replace digits with subscript format
    return re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', variable_name)


def causal_criteria(list1, list2):

    n1, n2 = np.count_nonzero(list1), np.count_nonzero(list2)
    c1, c2 = n1/len(list1), n2/len(list2)
    return [c1, c2]

def evaluate(actual, predicted):

    y_true_flat = [item for sublist in actual for item in sublist]
    y_pred_flat = [item for sublist in predicted for item in sublist]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    # Extract TP, TN, FP, FN from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate (Sensitivity)
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Negative Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    # Create a dictionary to store metrics
    shd = calculate_shd(np.array(actual), np.array(predicted))
    metrics = {
        # 'TP': tp,
        # 'TN': tn,
        # 'FP': fp,
        # 'FN': fn,
        'TPR': tpr,
        'TNR': tnr,
        'FPR': fpr,
        'FNR': fnr,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Fscore': f1_score,
        'SHD': shd
    }
    
    return metrics

# Diebold-Mariano test function
def dm_test(err1, err2, h=1, crit="MSE"):
    
    # Compute the loss differentials (MSE differences)
    d_mse = err1 - err2

    # Compute the mean and variance of the loss differentials
    mean_d = np.mean(d_mse)
    var_d = np.var(d_mse, ddof=1)

    # Number of forecasts
    T = len(d_mse)

    # Compute the DM test statistic
    dm_stat = mean_d / np.sqrt((var_d / T) * (1 + 2 * np.sum([1 - i / T for i in range(1, T)])))

    # Compute the p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value


# Calculate average metrics
def calculate_average_metrics(performance_dicts):
    sums = {}
    count = len(performance_dicts)
    
    for metrics in performance_dicts.values():
        for key, value in metrics.items():
            if key not in sums:
                sums[key] = 0
            sums[key] += value
    
    avg_metrics = {key: value / count for key, value in sums.items()}
    return avg_metrics


def calculate_group_means(df, params):
    """
    Calculate the mean of variables in each group and return a new DataFrame.

    Parameters:
    df (pd.DataFrame): The original DataFrame with variables.
    params (dict): A dictionary containing group information.

    Returns:
    pd.DataFrame: A DataFrame with the mean of variables in each group.
    """
    # Create a new DataFrame to store the mean of each group
    group_means = pd.DataFrame()

    # Calculate the mean of each group and add it as a new column in group_means
    for group, (start, end) in params['groups'].items():
        group_means[group] = df.iloc[:, start:end].mean(axis=1)

    return group_means


def generate_group_dicts(num_nodes, num_groups):
    
    groups, groups_cc, groups_fs = {}, {}, {}
    groups_size, groups_size_cc, groups_size_fs = {}, {}, {}
    
    base_group_size = num_nodes // num_groups
    remainder = num_nodes % num_groups
    
    start_idx, start_idx_cc, start_idx_fs = 0, 0, 0
    
    for k in range(num_nodes):

        group_key =  group_key = f"g{k+1}"
        end_idx_fs = start_idx_fs + 1
        groups_fs[group_key] = [start_idx_fs, end_idx_fs]
     
        groups_size_fs[group_key] = [1]
        start_idx_fs = end_idx_fs

    for i in range(num_groups):
        group_key = f"g{i+1}"
        
        if i < remainder:
            group_size = base_group_size + 1
        else:
            group_size = base_group_size
        
        end_idx = start_idx + group_size
        end_idx_cc = start_idx_cc + 1

        groups[group_key] = [start_idx, end_idx]
        groups_cc[group_key] = [start_idx_cc, end_idx_cc]
        groups_size[group_key] = [group_size]
        groups_size_cc[group_key] = [1]
        
        start_idx = end_idx
        start_idx_cc = end_idx_cc
    
    result = {
        'group_num': num_groups,
        'group_num_fs': num_nodes,
        'groups': groups,
        'groups_cc': groups_cc,
        'groups_fs': groups_fs,
        'groups_size': groups_size,
        'groups_size_cc': groups_size_cc,
        'groups_size_fs': groups_size_fs
        }
    
    return result


def calculate_group_means(df, params):
    """
    Calculate the mean of variables in each group and return a new DataFrame.

    Parameters:
    df (pd.DataFrame): The original DataFrame with variables.
    params (dict): A dictionary containing group information.

    Returns:
    pd.DataFrame: A DataFrame with the mean of variables in each group.
    """
    # Create a new DataFrame to store the mean of each group
    group_means = pd.DataFrame()

    # Calculate the mean of each group and add it as a new column in group_means
    for group, (start, end) in params['groups'].items():
        group_means[group] = df.iloc[:, start:end].mean(axis=1)

    return group_means


# Calculate average metrics
def calculate_average_metrics(performance_dicts):
    sums = {}
    count = len(performance_dicts)
    
    for metrics in performance_dicts.values():
        for key, value in metrics.items():
            if key not in sums:
                sums[key] = 0
            sums[key] += value
    
    avg_metrics = {key: value / count for key, value in sums.items()}
    return avg_metrics


# Plot metrics
def plot_metrics(methods_performance_dict, plot_path, metric_name):
    fig, ax = plt.subplots()

    for method, performance_dicts in methods_performance_dict.items():
        x = sorted(performance_dicts.keys())
        y = [performance_dicts[param][metric_name] for param in x]
        
        ax.plot(x, y, marker='.', linestyle='-', label=f'{method}')
    
    ax.set_xticks(x)
    plt.xticks(fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    plt.xlabel('Groups', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    ax.set_yticks(np.arange(0, 1.10, 0.10))  # Set finer ticks
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=13, ncol=2)
    # plt.legend().remove()

    plt.tight_layout()
    
    rnd = random.randint(1, 9999)
    filename = pathlib.Path(plot_path) / f'{metric_name}_groups_{rnd}.pdf'
    plt.savefig(filename)  # Save the figure
    # plt.show()

    methods_performance_dict = convert_numpy_types(methods_performance_dict)
     # Save the data as JSON
    json_filename = pathlib.Path(plot_path) / f'{metric_name}_data_{rnd}.json'
    with open(json_filename, 'w') as json_file:
        json.dump(methods_performance_dict, json_file, indent=4)


    # Plot metrics
def plot_metrics_tier(methods_performance_dict, plot_path, metric_name):
    fig, ax = plt.subplots()

    for method, performance_dicts in methods_performance_dict.items():
        x = sorted(performance_dicts.keys())
        y = [performance_dicts[param][metric_name] for param in x]
        
        ax.plot(x, y, marker='.', linestyle='-', label=f'{method}')
    
    ax.set_xticks(x)
    plt.xticks(fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    plt.xlabel('Groups', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    ax.set_yticks(np.arange(0, 1.10, 0.10))  # Set finer ticks
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=13)
    # plt.legend().remove()
    
    rnd = random.randint(1, 9999)
    filename = pathlib.Path(plot_path) / f'{metric_name}_groups_{rnd}.pdf'
    plt.savefig(filename)  # Save the figure
    # plt.show()


# Helper to make JSON-serializable (recursively handles keys & values)
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Plot metrics
def plot_real_metrics(methods_performance_dict, plot_path, metric_name):
    fig, ax = plt.subplots()

    for method, performance_dicts in methods_performance_dict.items():
        x = sorted(performance_dicts.keys())
        y = [performance_dicts[param][metric_name] for param in x]
        
        ax.plot(x, y, marker='.', linestyle='-', label=f'{method}')
    
    ax.set_xticks(x)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks(np.arange(0, 1.10, 0.10))
    plt.xlabel('Groups', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(title="", loc="best", ncol=2, fontsize=13, framealpha=0.5)
    
    rnd = random.randint(1, 9999)
    filename = pathlib.Path(plot_path) / f'{metric_name}_regime_{rnd}.pdf'
    plt.savefig(filename)

    # Save the data as JSON with fully safe values
    json_filename = pathlib.Path(plot_path) / f'{metric_name}_data_real_{rnd}.json'
    safe_dict = make_json_safe(methods_performance_dict)
    with open(json_filename, 'w') as json_file:
        json.dump(safe_dict, json_file, indent=4)

def calculate_multi_group_cc(df, group_sizes, regularization=1e-3):
    group_indices = []
    start_idx = 0
    for size in group_sizes:
        group_indices.append(list(range(start_idx, start_idx + size)))
        start_idx += size

    group_data = []
    group_labels = []
    for i, group in enumerate(group_indices):
        if len(group) < 2:
            # Use the single variable as-is
            group_data.append(df.iloc[:, group].values)
            group_labels.append(f'Group{i+1}_CC')
        else:
            group_data.append(df.iloc[:, group].values)
            group_labels.append(f'Group{i+1}_CC')

    # Perform MCCA only on groups with 2 or more variables
    mcca_input = [g for g in group_data if g.shape[1] > 1]
    single_vars = [g for g in group_data if g.shape[1] == 1]
    single_labels = [label for g, label in zip(group_data, group_labels) if g.shape[1] == 1]
    multi_labels = [label for g, label in zip(group_data, group_labels) if g.shape[1] > 1]

    canonical_df = pd.DataFrame()

    if mcca_input:
        mcca = MCCA(regs=regularization)
        mcca.fit(mcca_input)
        canonical_vars = mcca.transform(mcca_input)
        for can_var, label in zip(canonical_vars, multi_labels):
            canonical_df[label] = can_var[:, 0]

    for g, label in zip(single_vars, single_labels):
        canonical_df[label] = g[:, 0]

    # Sort columns to maintain original group order
    canonical_df = canonical_df[sorted(canonical_df.columns, key=lambda x: int(x.split('Group')[1].split('_')[0]))]

    return canonical_df



def generate_variable_list(N):
    """
    Generates a list of variable names in the format Var$_i$ where i ranges from 1 to N.
    
    Parameters:
    N (int): The number of variables to generate.
    
    Returns:
    list: A list of variable names formatted as Var$_i$.
    """
    return [f'Var$_{i}$' for i in range(1, N + 1)]



