from pathlib import Path

import pandas as pd

from mothernet.evaluation.cd_plot_new.cd_plot_code import cd_evaluation
from mothernet.evaluation.cd_plot_new.ni_plot_code import ni_evaluation


def load_data():
    df = pd.read_csv(Path(__file__).parent.parent / "results" / "results_test_2024-06-18.csv", index_col=0)
    model_rename = {
        "KNN": "KNN",
        "MLP": "MLP",
        "XGBoost": "XGBoost",
        "baam_nsamples500_numfeatures10_04_07_2024_17_04_53_epoch_1780": "GAMFormer (main effect)",
        "ebm_bins_main_effects": "EBM (main effect)",
        "RF": "Random Forest",
        "LogReg": "Logistic Regression",
        "TabPFN (Hollmann)": "TabPFN",
    }
    df.model = df.model.replace(model_rename)
    df = df[df.model.isin(model_rename.values())]

    df_slurm = pd.read_csv(Path(__file__).parent.parent / "results" / "results_slurm.csv")
    model_rename = {
        "ebm_default": "EBM (default)",
        # "ebm_bins": "EBM-bins (default)",
        "baam_0.1_pair_effect_fast": "GAMformer_fast-0.1",
        "baam_0.2_pair_effect_fast": "GAMformer_fast-0.2",
        # "baam_0.1_pair_effect": "GAMformer_greedy",
    }
    df_slurm.model = df_slurm.model.replace(model_rename)
    df_slurm = df_slurm[df_slurm.model.isin(model_rename.values())]

    # ['KNN', 'LogReg', 'MLP', 'MLP-Distill', 'MotherNet', 'RF', 'TabPFN (Hollmann)', 'XGBoost',
    #  'baam_binning_05_16_2024_16_16_27_epoch_2170', 'baam_embedding_binning_05_17_2024_00_02_36_epoch_1310',
    #  'baam_nfeatures_20_no_ensemble_e1520', 'baam_nsamples500_numfeatures10_04_07_2024_17_04_53_epoch_1780',
    #  'ebm_bins_main_effects', 'mn_Dclass_average_03_25_2024_17_14_32_epoch_2910_ohe_ensemble_8']

    return pd.concat([df, df_slurm], ignore_index=True)


def eval_am():
    input_data = load_data()
    # filter down to max time
    input_data = input_data[(input_data["max_time"] == 3600) | (input_data["best"].isna())]

    print(input_data.model.unique())

    # Take mean over folds
    input_data = input_data.groupby(["dataset", "model"]).mean(numeric_only=True).reset_index()

    print(input_data.model.unique())
    # Pivot for desired metric to create the performance per dataset table
    performance_per_dataset = input_data.pivot(index="dataset", columns="model", values="mean_metric")

    import matplotlib.pyplot as plt
    cd_evaluation(performance_per_dataset, maximize_metric=True, ignore_non_significance=False)
    plt.tight_layout()
    plt.savefig("cd_plot_pair.pdf")

    ni_evaluation(performance_per_dataset, maximize_metric=True, baseline_method="KNN")


if __name__ == "__main__":
    eval_am()
