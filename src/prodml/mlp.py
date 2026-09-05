import numpy as np
import torch
from torch import nn


class MLPWrapper:
    """PyTorch MLP exposing the sklearn fit/predict contract for the Trainer.

    Input is the sparse DictVectorizer matrix — densified per batch, never as
    a whole (the full dense matrix here is ~1.6 GB in float64).
    """

    def __init__(self, hidden_layer_size=64, epochs=10, lr=0.001, seed=42, batch_size=2048):
        self.hidden_layer_size = hidden_layer_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.batch_size = batch_size

    def fit(self, X, y):
        # Set the number of threads to 1 to avoid issues with multithreading in PyTorch
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        n, n_features = X.shape
        self._net = nn.Sequential(
            nn.Linear(n_features, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, 1),
        )
        y = np.asarray(y, dtype=np.float32)
        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)

        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                X_batch = torch.as_tensor(X[idx].toarray(), dtype=torch.float32)
                y_batch = torch.as_tensor(y[idx], dtype=torch.float32).view(-1, 1)
                opt.zero_grad()
                loss = loss_fn(self._net(X_batch), y_batch)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        self._net.eval()
        with torch.no_grad():
            if hasattr(X, "toarray"):
                preds = np.concatenate([
                    self._net(torch.as_tensor(X[start : start + self.batch_size].toarray(), dtype=torch.float32)).numpy()
                    for start in range(0, X.shape[0], self.batch_size)
                ])
            else:
                X = np.asarray(X, dtype=np.float32)
                preds = self._net(torch.as_tensor(X)).numpy()
        return preds.ravel()
