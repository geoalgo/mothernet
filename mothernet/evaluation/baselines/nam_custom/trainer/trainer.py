from types import SimpleNamespace
from typing import Sequence
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from mothernet.evaluation.baselines.nam_custom.config import Config
from mothernet.evaluation.baselines.nam_custom.trainer.losses import penalized_loss


class Trainer:

    def __init__(self, config: SimpleNamespace, model: Sequence[nn.Module],
                 dataloader: torch.utils.data.DataLoader, n_classes: int, verbose:
                 bool, epoch_callback: Optional[Callable]) -> None:
        self.config = Config(**vars(config))  #config
        self.model = model
        self.dataloader = dataloader
        self.n_classes = n_classes
        self.verbose = verbose
        self.epoch_callback = epoch_callback
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.lr,
                                          weight_decay=self.config.decay_rate)

        self.criterion = lambda inputs, targets, weights, fnns_out, model, classes: penalized_loss(
            self.config, inputs, targets, weights, fnns_out, model, classes
        )

    def train_step(self, model: nn.Module, optimizer: optim.Optimizer, batch: torch.Tensor) -> torch.Tensor:
        """Performs a single gradient-descent optimization step."""

        features, targets = batch
        #targets = targets.long()

        # Resets optimizer's gradients.
        self.optimizer.zero_grad()

        # Forward pass from the model.
        predictions, fnn_out = self.model(features)
        loss = self.criterion(predictions, targets, None, fnn_out, self.model,
                             self.n_classes)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
        try:
            self.model.train()
            for epoch in range(self.config.num_epochs):
                size = len(self.dataloader.dataset)
                losses = []
                for batch, (X, y) in enumerate(self.dataloader):
                    # Compute prediction and loss
                    loss = self.train_step(self.model, self.optimizer, (X, y))
                    losses.append(loss.item())

                if epoch % 10 == 0 and self.verbose:
                    loss, current = np.mean(losses), (batch + 1) * len(X)
                    print(f"epoch: {epoch}  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                if self.epoch_callback is not None:
                    self.epoch_callback(self.model, epoch, np.mean(losses))
        except KeyboardInterrupt:
            pass

