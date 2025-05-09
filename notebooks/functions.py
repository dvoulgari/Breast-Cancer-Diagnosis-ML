from scipy.stats import bootstrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_style("darkgrid")
sns.set_context("talk",  font_scale=0.6)

def process_results(all_results, classifiers):
    """
    Process evaluation results and calculate statistics for all classifiers.
    
    Args:
        all_results: Dictionary containing stored results from evaluation
        classifiers: List of classifier names that were evaluated
        
    Returns:
        Dictionary containing processed results and statistics for each classifier
    """
    # Initialize storage for processed results
    processed_results = {}
    
    # Combine results for each classifier
    for clf_name in classifiers:
        # Get all metrics DataFrames for this classifier
        clf_metrics = [res for res in all_results['metrics'] if res['Classifier'].iloc[0] == clf_name]
        
        if not clf_metrics:
            print(f"Warning: No results found for {clf_name}")
            continue
            
        # Combine all repetitions
        combined_df = pd.concat(clf_metrics)
        processed_results[clf_name] = {
            'all_metrics': combined_df,
            'statistics': {}
        }
        
        # Calculate statistics for each metric
        print(f"\n{'='*80}")
        print(f"{clf_name} - Performance Statistics")
        print(f"{'='*80}")
        
        # Select only numeric metric columns
        metric_cols = [col for col in combined_df.columns 
                      if col not in ['Classifier', 'Repetition', 'Fold', 'Best_Params'] 
                      and pd.api.types.is_numeric_dtype(combined_df[col])]
        
        for metric in metric_cols:
            data = combined_df[metric].dropna().values
            
            if len(data) == 0:
                print(f"{metric}: No valid data")
                continue
                
            # Calculate statistics
            median = np.median(data)
            mean = np.mean(data)
            std = np.std(data)
            
            # Calculate bootstrap CI
            try:
                bootstrap_ci = bootstrap(
                    (data,), 
                    np.median, 
                    confidence_level=0.95,
                    n_resamples=1000,
                    method='percentile'
                ).confidence_interval
                ci_low, ci_high = bootstrap_ci.low, bootstrap_ci.high
            except:
                ci_low, ci_high = np.nan, np.nan
            
            # Store statistics
            processed_results[clf_name]['statistics'][metric] = {
                'median': median,
                'mean': mean,
                'std': std,
                'ci_low': ci_low,
                'ci_high': ci_high
            }
            
            # Print results
            print(f"\n{metric}:")
            print(f"  Mean ± std: {mean:.3f} ± {std:.3f}")
            print(f"  Median: {median:.3f}")
            print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
            print(f"  N: {len(data)}")
    
    return processed_results

def compare_classifiers(processed_results, primary_metric='F2'):
    """
    Compare classifiers and select the best performing one based on median metrics.
    
    Args:
        processed_results: Dictionary from process_results() containing all metrics and statistics
        primary_metric: The metric to use for final ranking (default: F2)
        
    Returns:
        DataFrame with comparison results, sorted by primary metric
        Name of the best classifier
    """
    # Create comparison DataFrame
    comparison_data = []
    
    for clf_name, results in processed_results.items():
        # Get median values for all metrics
        stats = results['statistics']
        median_scores = {metric: vals['median'] for metric, vals in stats.items()}
        median_scores['Classifier'] = clf_name
        comparison_data.append(median_scores)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Handle case where primary metric might not exist
    available_metrics = comparison_df.columns.tolist()
    if primary_metric not in available_metrics:
        print(f"Warning: Primary metric '{primary_metric}' not found. Using first available metric.")
        primary_metric = available_metrics[0]
    
    # Sort by primary metric and reset index
    comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
    comparison_df = comparison_df.reset_index(drop=True)
    
    # Select best classifier
    best_classifier = comparison_df.iloc[0]['Classifier']
    
    # Print results
    print("\n" + "="*80)
    print("Final Classifier Comparison (Median Metrics)")
    print("="*80)
    print(comparison_df.to_string(float_format="%.3f"))
    print("\n" + "="*80)
    print(f"Best Classifier by {primary_metric}: {best_classifier}")
    print("="*80)
    
    return comparison_df, best_classifier

def plot_individual_metrics(comparison_df, processed_results):
    """
    Generate 4 plots for F1, F2, MCC, and Balanced Accuracy metrics.
    
    Args:
        comparison_df: DataFrame from compare_classifiers()
        processed_results: Dictionary from process_results()
    """
    metrics = ['F1', 'F2', 'MCC', 'Balanced_Accuracy']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        # Get data for each classifier
        means = []
        stds = []
        classifiers = []
        
        for clf_name in comparison_df['Classifier']:
            if metric in processed_results[clf_name]['statistics']:
                stats = processed_results[clf_name]['statistics'][metric]
                means.append(stats['mean'])
                stds.append(stats['std'])
                classifiers.append(clf_name)
        
        # Create bar plot
        bars = axes[i].bar(classifiers, means, yerr=stds, 
                         capsize=5, alpha=0.7,
                         color=sns.color_palette("husl", len(classifiers)))
        
        axes[i].set_title(f'{metric} Score Comparison', fontsize=14)
        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mean:.3f}±{std:.3f}',
                        ha='center', va='bottom', fontsize=10)
        
        # Set consistent y-axis limits
        axes[i].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('individual_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_best_classifier_dashboard(processed_results, best_classifier):
    """
    Generate comprehensive dashboard for the best classifier.
    
    Args:
        processed_results: Dictionary from process_results()
        best_classifier: Name of the best classifier
    """
    metrics = ['F1', 'F2', 'MCC', 'Balanced_Accuracy', 
               'Precision', 'Recall', 'ROC_AUC', 'Specificity']
    
    # Get all fold results for best classifier
    best_data = processed_results[best_classifier]['all_metrics']
    
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Distribution of all metrics
    plt.subplot(2, 2, 1)
    melted_df = best_data.melt(id_vars=['Classifier'], 
                             value_vars=metrics,
                             var_name='Metric', 
                             value_name='Score')
    sns.boxplot(data=melted_df, x='Metric', y='Score')
    plt.title(f'{best_classifier} - Metric Distributions', fontsize=14)
    plt.xticks(rotation=45)
    
    # Plot 3: Precision-Recall tradeoff
    plt.subplot(2, 2, 3)
    plt.scatter(best_data['Recall'], best_data['Precision'], alpha=0.6)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Tradeoff', fontsize=14)
    
    # Plot 4: ROC AUC vs F2
    plt.subplot(2, 2, 4)
    plt.scatter(best_data['ROC_AUC'], best_data['F2'], alpha=0.6)
    plt.xlabel('ROC AUC')
    plt.ylabel('F2 Score')
    plt.title('ROC AUC vs F2 Score', fontsize=14)
    
    plt.suptitle(f'Performance Analysis: {best_classifier}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{best_classifier}_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Create a clean confusion matrix without grid lines.
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar=False,
                    linewidths=0,  # Removes white grid lines
                    linecolor='none')  # Ensures no borders
    
    # Customize labels
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()