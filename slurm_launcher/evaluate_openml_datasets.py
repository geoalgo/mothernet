import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import scienceplots  # noqa
import wandb
from pygam import LinearGAM, LogisticGAM
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from mothernet.prediction.mothernet_additive import MotherNetAdditiveClassifierPairEffects
from mothernet.utils import get_mn_model
# from slurm_launcher.show_results import analyze_results

plt.style.use(['science', 'no-latex', 'light'])
plt.rcParams["figure.constrained_layout.use"] = True


class PyGAMSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, dataset_type, rs_samples=5):
        self.dataset_type = dataset_type
        self.rs_samples = rs_samples
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.dataset_type == "classification":
            gam = LogisticGAM()
            self.classes_ = np.unique(y)
        else:
            gam = LinearGAM()

        lams = np.random.rand(self.rs_samples, X.shape[1])
        lams = lams * 8 - 4
        lams = 10 ** lams

        self.model = gam.gridsearch(X, y, lam=lams)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if self.dataset_type == "classification":
            proba = self.model.predict_proba(X)
            if proba.ndim == 1:
                return np.column_stack((1 - proba, proba))
            return proba
        else:
            # For regression, return a dummy probability
            y_pred = self.model.predict(X)
            return np.column_stack((1 - y_pred, y_pred))

    def predict(self, X):
        check_is_fitted(self)
        if self.dataset_type == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        else:
            return self.model.predict(X)

    def get_params(self, deep=True):
        return {"dataset_type": self.dataset_type, "rs_samples": self.rs_samples}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def load_openml_data(task_id):
    try:
        task = openml.tasks.get_task(task_id, download_splits=True)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )

        # Drop rows with NaN values
        data = pd.concat([X, y], axis=1)
        data.dropna(inplace=True)

        X = data.drop(columns=[dataset.default_target_attribute])
        y = data[dataset.default_target_attribute]

        # Encode categorical target variable
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Split the data
        train_indices, test_indices = task.get_train_test_split_indices()
        train_mask = X.index.isin(train_indices)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]

        return {
            'problem': 'classification',
            'full': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
            'target_encoder': le,
            'dataset_name': dataset.name,
            'num_features': X.shape[1],
            'num_instances': X.shape[0]
        }
    except Exception as e:
        print(f"Error loading dataset {task_id}: {str(e)}")
        return None


def random_search_optimization(clf, param_dist, X, y, n_iter=100, cv=5):
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring='roc_auc',
                                       n_jobs=-1, random_state=42)
    random_search.fit(X, y)
    return random_search.best_estimator_


def process_model(model, model_name, X, y, X_test, y_test, n_splits=5, n_jobs=1, record_shape_functions=False,
                  column_names=None):
    start_time = time.time()
    cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring='roc_auc', n_jobs=n_jobs)
    model.fit(X, y)
    end_time = time.time()

    y_pred = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred)

    record = {
        'model': model_name,
        'cv_auc_mean': np.mean(cv_scores),
        'cv_auc_std': np.std(cv_scores),
        'test_auc': test_auc,
        'fit_time': end_time - start_time
    }

    if record_shape_functions and hasattr(model, 'named_steps') and 'baam' in model.named_steps:
        baam = model.named_steps['baam']
        record['shape_functions'] = {
            'bin_edges': baam.bin_edges_,
            'w': baam.w_,
            'column_names': column_names
        }

    return record


def benchmark_models(dataset_info, models, ct=None, n_splits=3, **wandb_kwargs):
    if dataset_info is None:
        return None

    X, y = dataset_info['full']['X'], dataset_info['full']['y']
    X_test, y_test = dataset_info['test']['X'], dataset_info['test']['y']
    dataset_name = dataset_info['dataset_name']
    column_names = X.columns

    if ct is None:
        is_cat = np.array([dt.kind == 'O' for dt in X.dtypes])
        cat_cols = X.columns.values[is_cat]
        num_cols = X.columns.values[~is_cat]

        cat_ohe_step = ('ohe', OneHotEncoder(handle_unknown='ignore'))

        cat_pipe = Pipeline([cat_ohe_step])
        num_pipe = Pipeline([('identity', FunctionTransformer())])
        transformers = [
            ('cat', cat_pipe, cat_cols),
            ('num', num_pipe, num_cols)
        ]
        ct = ColumnTransformer(transformers=transformers, sparse_threshold=0)

    records = []
    summary_record = {
        'dataset_name': dataset_name,
        'num_features': dataset_info['num_features'],
        'num_instances': dataset_info['num_instances']
    }

    models_pipelines = [
        (model, pipeline_fun(X, y, ct)) for model, pipeline_fun in models.items()
    ]

    for model_name, model in models_pipelines:
        try:
            wandb.init(
                name=dataset_name,
                config={
                    "n_pair_feature_max_ratio": model[-1].n_pair_feature_max_ratio if hasattr(model[-1], 'n_pair_feature_max_ratio') else 0,
                    "dataset_name": dataset_name,
                },
                **wandb_kwargs
            )
            record = process_model(model, model_name, X, y, X_test, y_test, n_splits=n_splits, n_jobs=1,
                                   record_shape_functions=False, column_names=column_names)
            record.update(summary_record)

            wandb.log(record)
            wandb.finish()
            records.append(record)
        except Exception as e:
            print(f"Error processing model {model_name} for dataset {dataset_name}: {str(e)}")
            raise e

    return records


def evaluate_models_on_openml(task_ids, models, n_splits: int = 5, **wandb_kwargs):
    all_results = []
    dataset_stats = []

    for task_id in tqdm(task_ids):
        print(f"Evaluating task ID: {task_id}")

        # Load the OpenML dataset
        dataset_info = load_openml_data(task_id)

        if dataset_info:
            dataset_stats.append({
                'task_id': task_id,
                'dataset_name': dataset_info['dataset_name'],
                'num_features': dataset_info['num_features'],
                'num_instances': dataset_info['num_instances']
            })
            print(dataset_stats[-1])

            # Perform the evaluation
            results = benchmark_models(dataset_info, models, n_splits=n_splits, **wandb_kwargs)

            if results:
                all_results.extend(results)

    # Prepare the final JSON structure
    final_results = {
        'dataset_statistics': dataset_stats,
        'model_results': all_results
    }

    # Save all results as a single JSON file
    time_stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    os.makedirs("results", exist_ok=True)
    with open(f"results/openml_results_{time_stamp}.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"All results saved in results/openml_results_{time_stamp}.json")
    return f"results/openml_results_{time_stamp}.json"



if __name__ == '__main__':
    # start a new wandb run to track this script

    task_ids = [
        15,
        9951,
        272,
        39,
        31,
        10093,
        57,
        37,
        # Too large for now in term of number of rows
        43,
        # 24,
        9952,
        # 310,
        # 3483,
        # large datasets
        14968,
        9969,
        9953,
    ]
    baam_model_string = "baam_nsamples500_numfeatures10_04_07_2024_17_04_53_epoch_1780.cpkt"
    n_pair_feature_max_ratio = os.getenv("n_pair_feature_max_ratio", default="0.9")
    n_pair_feature_max_ratio = float(n_pair_feature_max_ratio)
    print(f"Evaluating with {n_pair_feature_max_ratio} top pairs.")

    models = {
        f'baam_{n_pair_feature_max_ratio}pairs': lambda X, y, ct: Pipeline([
            ('ct', ct),
            ('baam_pair', MotherNetAdditiveClassifierPairEffects(
                device='cpu',
                path=get_mn_model(baam_model_string),
                n_pair_feature_max_ratio=n_pair_feature_max_ratio
            )),
        ]),
        # 'lr': lambda X, y, ct: Pipeline([
        #     ('ct', ct),
        #     ('std', StandardScaler()),
        #     ('lr', LogisticRegression(random_state=0)),
        # ]),
        # 'ebm_optimized': lambda X, y, ct: random_search_optimization(
        #     Pipeline([
        #         ('ct', ct),
        #         ('ebm', ExplainableBoostingClassifier(n_jobs=-1, random_state=0))
        #     ]),
        #     {
        #         'ebm__max_rounds': randint(100, 500),
        #         'ebm__max_bins': randint(32, 256),
        #         'ebm__max_interaction_bins': randint(32, 256),
        #         'ebm__learning_rate': uniform(0.01, 0.3),
        #         'ebm__interactions': randint(0, 10)
        #     },
        #     X, y, n_iter=10
        # ),
        # 'ebm_main_effects_optimized': lambda X, y, ct: random_search_optimization(
        #     Pipeline([
        #         ('ct', ct),
        #         ('ebm', ExplainableBoostingClassifier(n_jobs=-1, random_state=0, interactions=0))
        #     ]),
        #     {
        #         'ebm__max_rounds': randint(100, 500),
        #         'ebm__max_bins': randint(32, 256),
        #         'ebm__max_interaction_bins': randint(32, 256),
        #         'ebm__learning_rate': uniform(0.01, 0.3)
        #     },
        #     X, y, n_iter=10
        # ),
        # 'baam': lambda X, y, ct: Pipeline([
        #     ('ct', ct),
        #     ('baam', MotherNetAdditiveClassifier(device='cpu', path=get_mn_model(baam_model_string))),
        # ])
    }
    wandb_kwargs = dict(
        project="Gaamformer",
        group="pair-effect-v1",
        tags=["all-tasks-v1"],
    )
    result_file = evaluate_models_on_openml(task_ids, models, n_splits=5, **wandb_kwargs)
