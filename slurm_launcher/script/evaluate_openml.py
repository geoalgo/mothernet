import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from mothernet.utils import get_mn_model

warnings.simplefilter("ignore", FutureWarning)  # openml deprecation of array return type
from mothernet.datasets import load_openml_list, open_cc_dids
from mothernet.evaluation.baselines.tabular_baselines import knn_metric, logistic_metric, xgb_metric, random_forest_metric, mlp_metric
from mothernet.evaluation.tabular_evaluation import transformer_metric, is_classification
from mothernet.evaluation import tabular_metrics
from mothernet.prediction.tabpfn import TabPFNClassifier
from functools import partial
from mothernet.evaluation.tabular_evaluation import eval_on_datasets
from mothernet.prediction.mothernet_additive import MotherNetAdditiveClassifier, MotherNetAdditiveClassifierPairEffects, \
    MotherNetAdditiveClassifierPairEffectsAuto

from interpret.glassbox import ExplainableBoostingClassifier


from sklearn import set_config

pair_ratios = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 0.9]
allowed_methods = [
    'knn',
    'rf_new_params',
    'xgb',
    'logistic',
    # 'mlp',
    'tabpfn',
    'ebm_default',
    'ebm_bins',
    'ebm_bins_main_effects',
    'baam_single_effect',
    'baam_auto_pair_effect_fast',
]
for pair_ratio in pair_ratios:
    # allowed_methods.append(f"baam_{pair_ratio}_pair_effect")
    allowed_methods.append(f"baam_{pair_ratio}_pair_effect_fast")


def evaluate(method: str | None, n_datasets: int | None):
    assert method is None or method in allowed_methods
    if n_datasets is None:
        n_datasets = len(open_cc_dids)
    cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(open_cc_dids[:n_datasets], multiclass=True, shuffled=True, filter_for_nan=False, max_samples = 10000, num_feats=100, return_capped=True)
    eval_positions = [1000]
    max_features = 100
    n_samples = 2000
    base_path = Path("./")
    overwrite = False
    # max_times = [1, 15, 30, 60, 60 * 5, 60 * 15, 60*60]
    max_times = [60 * 60]
    metric_used = tabular_metrics.auc_metric
    split_numbers = [1, 2, 3, 4, 5]
    task_type = 'multiclass'

    (base_path / "results").mkdir(exist_ok=True, parents=True)
    (base_path / "results" / "tabular").mkdir(exist_ok=True, parents=True)
    (base_path / "results" / "tabular" / "multiclass").mkdir(exist_ok=True, parents=True)

    cc_test_datasets_multiclass_df['isNumeric'] = (cc_test_datasets_multiclass_df.NumberOfSymbolicFeatures == 1) & (cc_test_datasets_multiclass_df.NumberOfInstancesWithMissingValues == 0)
    cc_test_datasets_multiclass_df['NumberOfInstances'] = cc_test_datasets_multiclass_df['NumberOfInstances'].astype(int)
    cc_test_datasets_multiclass_df['NumberOfFeatures'] = cc_test_datasets_multiclass_df['NumberOfFeatures'].astype(int)
    cc_test_datasets_multiclass_df['NumberOfClasses'] = cc_test_datasets_multiclass_df['NumberOfClasses'].astype(int)

    # print(cc_test_datasets_multiclass_df[['did', 'name', 'NumberOfFeatures', 'NumberOfInstances', 'NumberOfClasses']].rename(columns={'NumberOfFeatures': "d", "NumberOfInstances":"n", "NumberOfClasses": "k"}).to_latex(index=False))

    clf_dict = {
        'knn': knn_metric,
        'rf_new_params': random_forest_metric,
        'xgb': xgb_metric,
        'logistic': logistic_metric,
        'mlp': mlp_metric,
    }

    for model_name, model in clf_dict.items():
        if method is None or model_name == method:
            results_baselines = [
                eval_on_datasets(
                    'multiclass', model, model_name, cc_test_datasets_multiclass, eval_positions=eval_positions, max_times=max_times,
                                 metric_used=metric_used, split_numbers=split_numbers,
                                 n_samples=n_samples, base_path=base_path, fetch_only=False)
            ]
            save_results(results=results_baselines, name=f"baselines-{model_name}")

    set_config(skip_parameter_validation=True, assume_finite=True)

    # transformers don't have max times
    max_times = [1]
    # device = "cuda:1"
    device = "cpu"

    model_string = "tabpfn_nooptimizer_emsize_512_nlayers_12_steps_2048_bs_32ada_lr_0.0001_1_gpu_07_24_2023_01_43_33"
    tabpfn_ours = TabPFNClassifier(device=device, model_string=model_string, epoch="1650", N_ensemble_configurations=3)
    ebm_bins_main_effects = ExplainableBoostingClassifier(max_bins=64, interactions=0)
    baam_model_string = "baam_nsamples500_numfeatures10_04_07_2024_17_04_53_epoch_1780.cpkt"
    clf_dict = {
        'tabpfn': transformer_metric,
        'ebm_default': partial(transformer_metric, classifier=ExplainableBoostingClassifier()),
        'ebm_bins': partial(transformer_metric, classifier=ExplainableBoostingClassifier(max_bins=64)),
        'ebm_bins_main_effects': partial(transformer_metric, classifier=ExplainableBoostingClassifier(max_bins=64, interactions=0)),
        'baam_single_effect': MotherNetAdditiveClassifier(
            path=get_mn_model(baam_model_string), device=device
        ),
    }


    for pair_ratio in pair_ratios:
        # clf_dict[f"baam_{pair_ratio}_pair_effect"] = partial(transformer_metric, MotherNetAdditiveClassifierPairEffects(
        #     device='cpu',
        #     path=get_mn_model(baam_model_string),
        #     n_pair_feature_max_ratio=pair_ratio,
        # )),
        clf_dict[f"baam_auto_pair_effect_fast"] = MotherNetAdditiveClassifierPairEffectsAuto(
            device='cpu',
            path=get_mn_model(baam_model_string),
            pair_strategy="fast",
        )

        clf_dict[f"baam_{pair_ratio}_pair_effect_fast"] = MotherNetAdditiveClassifierPairEffects(
            device='cpu',
            path=get_mn_model(baam_model_string),
            n_pair_feature_max_ratio=pair_ratio,
            pair_strategy="fast",
        )

    for model_name, model in clf_dict.items():
        if method is None or model_name == method:
            results_transformers = [
                eval_on_datasets('multiclass', model, model_name, cc_test_datasets_multiclass, eval_positions=eval_positions,
                                 max_times=max_times,
                                 metric_used=metric_used, split_numbers=split_numbers,
                                 n_samples=n_samples, base_path=base_path, overwrite=True, device=device)
            ]
            save_results(results=results_transformers, name=f"transformers-{model_name}")

    # save_results(results=results_baselines + results_transformers, name="all")


def save_results(results, name: str):
    flat_results = []
    for per_dataset in results:
        for result in per_dataset:
            row = {}
            for key in ['dataset', 'model', 'mean_metric', 'split', 'max_time']:
                row[key] = result[key]
            best_configs_key, = [k for k in result.keys() if "best_configs" in k]
            if result[best_configs_key][0] is not None:
                row.update(result[best_configs_key][0])
            row['mean_metric'] = float(row["mean_metric"].numpy())
            flat_results.append(row)

    results_df = pd.DataFrame(flat_results)
    results_df['model'] = results_df.model.replace({
        'knn': "KNN",
        'rf_new_params': 'RF',
        'mlp': "MLP",
        'xgb':'XGBoost',
        'logistic': 'LogReg',
        'tabpfn': 'TabPFN (Hollmann)',
        # 'mlp_distill': 'MLP-Distill',
        # 'mothernet': 'MotherNet',
        # 'tabpfn_ours': 'TabPFN (ours)'
    })

    filename = f"results/results_test_{name}_{datetime.today()}.csv"
    results_df.to_csv(filename)
    print(filename)

@dataclass
class Args:
    method: str
    n_datasets: int | None = None

    @classmethod
    def parse(cls, parser):
        # experiment setup arguments
        parser.add_argument(
            '--method',
            type=str,
        )
        parser.add_argument(
            '--n_datasets',
            type=int,
        )
        args = parser.parse_args()
        return Args(**args.__dict__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate OpenML datasets for GAMFormer paper.',
    )
    args = Args.parse(parser)
    print(args)
    assert args.method is None or args.method in allowed_methods
    evaluate(
        method=args.method,
        n_datasets=args.n_datasets
    )
