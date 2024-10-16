import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from mothernet.evaluation.baselines.nam_custom.config import defaults
from mothernet.evaluation.baselines.nam_custom.data import NAMDataset
from mothernet.evaluation.baselines.nam_custom.models import NAM, get_num_units
from mothernet.evaluation.baselines.nam_custom.trainer import Trainer


_DEFAULT_CONFIG = defaults()


def _encode_y(y):
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()
    if y.ndim == 1:
        le = LabelEncoder()
        y = le.fit_transform(y)
        classes = le.classes_
    else:
        # used probabilities as labels
        classes = torch.arange(y.shape[1])
    return y, classes


class TorchNAM(ClassifierMixin, BaseEstimator):
    def __init__(self, n_epochs=10, lr=0.01, output_regularization=0.0,
                 dropout=0.5, feature_dropout=0.0, hidden_sizes=[],
                 weight_decay=0.01, batch_size=1024, verbose=0, device='cuda',
                 epoch_callback=None, nonlinearity='exu', init_state=None,
                 regression=False):
        self.verbose = verbose
        self.device = device
        self.epoch_callback = epoch_callback
        self.init_state = init_state
        self.regression = regression
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.weight_decay = weight_decay
        self.nonlinearity = nonlinearity
        self.output_regularization = output_regularization

        self.config = _DEFAULT_CONFIG
        self.config.device = device
        self.config.regression = regression
        self.config.batch_size = batch_size
        self.config.lr = lr
        self.config.num_epochs = n_epochs
        self.config.activation = nonlinearity
        self.config.l2_regularization = weight_decay
        self.config.output_regularization = output_regularization
        self.config.dropout = dropout
        self.config.feature_dropout = feature_dropout
        self.config.hidden_sizes = hidden_sizes

    def make_model(self, n_features, num_units, n_classes):
        return NAM(config=self.config, name='NAM', num_inputs=n_features,
                   num_units=num_units, n_classes=n_classes)

    def fit_from_dataloader(self, dataloader, n_features, classes):
        n_classes = len(classes)
        features = dataloader.dataset.tensors[0]
        y = dataloader.dataset.tensors[1]

        x1 = pd.DataFrame(features.cpu().numpy(), columns=[f'x{i}' for i in range(n_features)])
        y1 = pd.DataFrame(y.cpu().numpy(), columns=['y'])
        data = pd.concat([x1, y1], axis=1)

        dataset = NAMDataset(self.config, data_path=data,
                             features_columns=data.columns[:-1],
                             targets_column=data.columns[-1])

        model = self.make_model(len(dataset[0][0]), get_num_units(self.config,
                                                                  dataset.features),
                               n_classes=n_classes)
        model = model.to(self.device)

        # loading the state dict seems the easiest way to ensure all the configs actually match
        if self.init_state is not None:
            model.load_state_dict(self.init_state)

        trainer = Trainer(self.config, model, dataloader, n_classes,
                          self.verbose, self.epoch_callback)
        trainer.train()

        self.model_ = model
        self.classes_ = classes

    def fit(self, X, y):
        y, classes = _encode_y(y)
        if torch.is_tensor(X):
            X = X.clone().detach().to(self.device).float()
        else:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if torch.is_tensor(y):
            y = y.clone().detach().to(self.device)
        else:
            y = torch.tensor(y, device=self.device)
        X = X.nan_to_num()
        dataloader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size or X.shape[0])
        self.fit_from_dataloader(dataloader, n_features=X.shape[1], classes=classes)
        return self

    def _predict(self, X):
        if torch.is_tensor(X):
            X = X.clone().detach().to(self.device).float()
        else:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.model_(X.nan_to_num())[0]

    def predict(self, X):
        pred = self._predict(X)
        return self.classes_[pred.argmax(1).detach().cpu().numpy()]

    def predict_proba(self, X):
        pred = self._predict(X)
        return pred.softmax(dim=1).detach().cpu().numpy()



