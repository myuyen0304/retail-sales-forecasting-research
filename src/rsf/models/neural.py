"""RNN và LSTM (PyTorch, chạy được trên CPU).

Bản chính (calendar-aware, không tự hồi quy): đầu vào là chuỗi biến lịch của L ngày kết thúc tại t,
đầu ra log y_t. Biến lịch của tương lai đã biết, nên dự báo 549 ngày không dồn sai số.
Bản `_ar` (pure, ablation): đầu vào là L giá trị log y gần nhất, dự báo đệ quy (cách của pilot cũ).

Số epoch chọn bằng early stopping trên 10% cuối của train, rồi fit lại trên toàn train với số epoch đó.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from rsf.models.base import Forecaster

DEFAULTS = dict(lookback=28, hidden=32, layers=1, lr=1e-3, batch=64, max_epochs=300, patience=25,
                holdout=0.1, dropout=0.0)
# Số luồng torch cho mỗi tiến trình. Laptop 12 lõi, --jobs 3: để 4. Colab 2 lõi, --jobs 2: đặt 1.
THREADS = int(os.environ.get("RSF_TORCH_THREADS", 4))


def _net(cell: str, n_in: int, hidden: int, layers: int, dropout: float):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            rnn = nn.LSTM if cell == "lstm" else nn.RNN
            self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return Net()


def _train(cell, X, y, p, seed, epochs=None):
    """Trả (net, số epoch tốt nhất). epochs=None: early stopping trên đoạn cuối; ngược lại train đúng số epoch."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(THREADS)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    if epochs is None:
        cut = int(len(X) * (1 - p["holdout"]))
        tr, va = slice(0, cut), slice(cut, len(X))
    else:
        tr, va = slice(0, len(X)), None
    net = _net(cell, X.shape[2], p["hidden"], p["layers"], p["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
    loss_fn = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    n_tr = tr.stop - tr.start
    best, best_ep, best_state, wait = np.inf, 0, None, 0
    for ep in range(1, (epochs or p["max_epochs"]) + 1):
        net.train()
        for idx in torch.randperm(n_tr, generator=g).split(p["batch"]):
            opt.zero_grad()
            loss_fn(net(Xt[tr][idx]), yt[tr][idx]).backward()
            opt.step()
        if va is None:
            continue
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(Xt[va]), yt[va]))
        if v < best - 1e-6:
            best, best_ep, wait = v, ep, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= p["patience"]:
                break
    if va is not None:
        net.load_state_dict(best_state)
    return net, (best_ep if va is not None else epochs)


def _predict(net, X):
    import torch
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy()


class _NeuralCal(Forecaster):
    family = "dl"
    calendar_aware = True
    stochastic = True
    cell = "lstm"

    def _windows(self, dates):
        L = self.p["lookback"]
        d = pd.DatetimeIndex(dates)
        full = pd.date_range(d[0] - pd.Timedelta(days=L - 1), d[-1], freq="D")
        F = ((self.cal.features(full, "tree")[self.columns] - self.mu) / self.sd).values
        pos = full.get_indexer(d)
        return np.stack([F[i - L + 1:i + 1] for i in pos])

    def fit(self, y, cal):
        self.p = {**DEFAULTS, **self.params}
        self.cal, self.index = cal, y.index
        F = cal.features(y.index, "tree")
        self.columns = list(F.columns)
        self.mu, self.sd = F.mean(), F.std().replace(0, 1.0)
        ly = np.log(y.values)
        self.ym, self.ys = ly.mean(), ly.std()
        X = self._windows(y.index)
        z = (ly - self.ym) / self.ys
        _, self.epochs = _train(self.cell, X, z, self.p, self.seed)
        self.net, _ = _train(self.cell, X, z, self.p, self.seed, epochs=max(self.epochs, 1))
        return self

    def predict(self, dates):
        return np.exp(_predict(self.net, self._windows(dates)) * self.ys + self.ym)

    def fitted(self):
        return pd.Series(self.predict(self.index), index=self.index)


class _NeuralAR(Forecaster):
    family = "dl"
    stochastic = True
    cell = "lstm"

    def fit(self, y, cal):
        self.p = {**DEFAULTS, "lookback": 56, **self.params}
        L = self.p["lookback"]
        self.index = y.index
        ly = np.log(y.values)
        self.ym, self.ys = ly.mean(), ly.std()
        z = (ly - self.ym) / self.ys
        X = np.stack([z[t - L:t] for t in range(L, len(z))])[:, :, None]
        _, self.epochs = _train(self.cell, X, z[L:], self.p, self.seed)
        self.net, _ = _train(self.cell, X, z[L:], self.p, self.seed, epochs=max(self.epochs, 1))
        self.hist = list(z)
        self._fitted = pd.Series(np.exp(_predict(self.net, X) * self.ys + self.ym), index=y.index[L:])
        return self

    def predict(self, dates):
        L, hist, out = self.p["lookback"], list(self.hist), []
        for _ in dates:
            v = float(_predict(self.net, np.array(hist[-L:], dtype=float)[None, :, None])[0])
            hist.append(v)
            out.append(v)
        return np.exp(np.asarray(out) * self.ys + self.ym)

    def fitted(self):
        return self._fitted


RNN = type("RNN", (_NeuralCal,), {"name": "rnn", "cell": "rnn"})
LSTM = type("LSTM", (_NeuralCal,), {"name": "lstm", "cell": "lstm"})
RNN_AR = type("RNN_AR", (_NeuralAR,), {"name": "rnn_ar", "cell": "rnn"})
LSTM_AR = type("LSTM_AR", (_NeuralAR,), {"name": "lstm_ar", "cell": "lstm"})
