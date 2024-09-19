import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

root_result_dir = Path(__file__).parent

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def create_summary_dataframe(results):
    model_results = results['model_results']
    df = pd.DataFrame(model_results)
    return df


def calculate_average_performance(df):
    return df.groupby('model')[['cv_auc_mean', 'test_auc', 'fit_time']].mean()


def perform_statistical_tests(df):
    models = df['model'].unique()
    test_results = {}

    for metric in ['cv_auc_mean', 'test_auc']:
        test_results[metric] = {}
        for model1 in models:
            for model2 in models:
                if model1 != model2:
                    scores1 = df[df['model'] == model1][metric]
                    scores2 = df[df['model'] == model2][metric]
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    test_results[metric][f"{model1} vs {model2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value
                    }

    return test_results

def figure_dir():
    res = root_result_dir / "figures"
    res.mkdir(exist_ok=True)
    return res

def plot_performance_comparison(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='cv_auc_mean', data=df)
    sns.stripplot(x='model', y='cv_auc_mean', data=df)
    plt.title('Cross-Validation AUC Scores by Model')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figure_dir() / 'cv_auc_comparison.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='test_auc', data=df)
    plt.title('Test AUC Scores by Model')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figure_dir() / 'test_auc_comparison.png')
    plt.close()


def plot_fit_time_comparison(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='fit_time', data=df)
    plt.title('Fit Time by Model')
    plt.ylabel('Fit Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figure_dir() / 'fit_time_comparison.png')
    plt.close()


def create_dataset_size_auc_plot(df, metric='cv_auc_mean'):
    # Create the scatter plot
    plt.figure(figsize=(12, 8))

    models = df['model'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']  # Add more markers if needed

    for model, color, marker in zip(models, colors, markers):
        model_data = df[df['model'] == model]
        plt.scatter(model_data['num_instances'], model_data[metric],
                    label=model, color=color, marker=marker, s=50, alpha=0.7)

    plt.xscale('log')  # Use log scale for dataset size
    plt.xlabel('Dataset Size (Number of Instances)')
    plt.ylabel('ROC AUC Score')
    plt.title(f'Model Performance vs Dataset Size based on {metric}')
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(figure_dir() / f'dataset_size_auc_plot_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_rank_plot(df, metric='cv_auc_mean'):
    print("Input DataFrame:")
    print(df.head())
    print("\nColumns:")
    print(df.columns)

    # Compute ranks for each dataset
    df_ranked = df.groupby('dataset_name')[['model', metric]].rank(ascending=False, method='min')

    print("\nRanked DataFrame:")
    print(df_ranked.head())
    print("\nRanked DataFrame Columns:")
    print(df_ranked.columns)

    # Reset index to make 'dataset_name' a column
    df_ranked = df_ranked.reset_index()

    # Rename the ranked metric column to 'rank'
    df_ranked = df_ranked.rename(columns={metric: 'rank'})

    print("\nFinal Ranked DataFrame:")
    print(df_ranked.head())
    print("\nFinal Ranked DataFrame Columns:")
    print(df_ranked.columns)

    # Merge ranks back to original dataframe
    df_with_ranks = pd.merge(df, df_ranked[['dataset_name', 'model', 'rank']], on=['dataset_name', 'model'])

    # Compute average ranks
    avg_ranks = df_with_ranks.groupby('model')['rank'].mean().sort_values()

    # Create the rank plot
    plt.figure(figsize=(12, 6))
    sns.boxenplot(x='model', y='rank', data=df_with_ranks, order=avg_ranks.index)
    plt.title(f'Model Ranking based on {metric}')
    plt.ylabel('Rank (lower is better)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.gca().invert_yaxis()  # Invert y-axis so that lower (better) ranks are on top

    # Add average rank annotations
    for i, model in enumerate(avg_ranks.index):
        plt.text(i, plt.gca().get_ylim()[0], f'{avg_ranks[model]:.2f}',
                 horizontalalignment='center', verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(figure_dir() / f'rank_plot_{metric}.png')
    plt.close()


def analyze_results(file_path, df_gaam):

    # Step 2: Create a summary dataframe
    df = pd.concat([
        create_summary_dataframe(load_results(file_path)),
        df_gaam
    ], ignore_index=True)

    rows = []
    for dataset in df.dataset_name.unique():
        df_dataset = df.loc[(df.dataset_name == dataset) & (df.model.str.contains("baam")), :]
        df_dataset = df_dataset.sort_values(by="cv_auc_mean", ascending=False)
        row = next(iter(df_dataset.to_dict(orient="records")))
        row["model"] = "baam-best-pair-CV"
        rows.append(row)

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=False)
    # Step 3: Calculate average performance
    avg_performance = calculate_average_performance(df)
    print("Average Performance:")
    print(avg_performance)
    avg_performance.to_csv("average_performance.csv")
    print("\n")

    # Step 4: Perform statistical tests
    stat_tests = perform_statistical_tests(df)
    print("Statistical Test Results:")
    for metric, tests in stat_tests.items():
        print(f"\n{metric.upper()}:")
        for comparison, result in tests.items():
            print(f"{comparison}: t-statistic = {result['t_statistic']:.4f}, p-value = {result['p_value']:.4f}")

    # Step 5: Create visualizations
    plot_performance_comparison(df)
    plot_fit_time_comparison(df)

    # Ensure 'dataset_name' is a column, not an index
    if 'dataset_name' not in df.columns and df.index.name == 'dataset_name':
        df = df.reset_index()

    # Create the new dataset size vs rank plot
    create_dataset_size_auc_plot(df, 'cv_auc_mean')
    create_dataset_size_auc_plot(df, 'test_auc')

    # Step 6: Create rank plots
    # create_rank_plot(df, 'cv_auc_mean')
    # create_rank_plot(df, 'test_auc')

    # Step 7: Analyze dataset impact
    # TODO make sure that dataset statistics match
    results = load_results(file_path)
    dataset_stats = pd.DataFrame(results['dataset_statistics'])
    correlation = df.groupby('dataset_name')['cv_auc_mean'].mean().corr(
        dataset_stats.set_index('dataset_name')['num_features'])
    print(f"\nCorrelation between number of features and average CV AUC: {correlation:.4f}")

import wandb

def load_wandb_results(path="geoalgo-university-of-freiburg/Gaamformer", tag="all-tasks-v1"):
    api = wandb.Api()
    runs = api.runs(path=path, filters={"tags": {"$in": [tag]}})
    rows = []
    for run in runs:
        print(run)
        if "n_pair_feature_max_ratio" in run.config:
            row = {
                "dataset_name": run.config["dataset_name"],
                "n_pair_feature_max_ratio": run.config["n_pair_feature_max_ratio"],
            }
            row.update(run.summary)
            rows.append(row)
    df_res = pd.DataFrame(rows).dropna()
    cols = [
        "dataset_name", "n_pair_feature_max_ratio", "cv_auc_mean", "cv_auc_std", "fit_time", "model", "num_features",
        "num_instances", "test_auc"
    ]
    return df_res.loc[:, cols]



if __name__ == '__main__':
    result_file = root_result_dir / "results/openml_results_09_16_2024_21_55_51_baselines.json"
    df_gaam = load_wandb_results().drop_duplicates(["dataset_name", "model"])
    analyze_results(result_file, df_gaam)

    df_avg = pd.read_csv("average_performance.csv")
    print(df_avg.loc[:, ["model", "cv_auc_mean", "test_auc"]].to_string(index=False))

    # print(df_avg.loc[df_avg.model.str.contains("[ebm*|lr]"), ["model", "test_auc", "fit_time"]].to_string())

    df_avg = df_avg[df_avg.model.str.contains("pairs")]
    df_avg["n_pairs"] = df_avg["model"].apply(lambda s: float(s.replace("pairs", "").replace("baam_", "")))
    print(df_avg.loc[:, ["n_pairs", "test_auc", "fit_time"]].sort_values(by="n_pairs").to_string(index=False))