"""Section 4.4: FFNN, RNN and LSTM trained on the pinball loss of the three quantiles at once.

A monotone output head keeps VaR_1% > VaR_2.5% > VaR_5% by construction (softplus base plus ReLU
spreads with a floor epsilon), and a spread penalty discourages quantile collapse. Inputs: ATM, RR25,
ATM x RR25, plus OVX for USD/CAD and the mini-budget dummy for GBP/USD, standardised on the training sample.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from g7var import CFG
from g7var.backtest import forecasts
from g7var.data import master, split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTRA = {"USD/CAD": ["OVX"], "GBP/USD": ["Mini_Budget_Crisis", "Crisis_x_BF25"]}
D = CFG["deep"]


class PinballLoss(nn.Module):
    """Mean pinball loss over the quantiles plus a penalty when neighbouring quantiles come closer than min_spread."""

    def __init__(self, quantiles, lam, min_spread):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))
        self.lam, self.min_spread = lam, min_spread

    def forward(self, pred, y):
        err = y.expand_as(pred) - pred
        pinball = torch.where(err >= 0, self.q * err, (self.q - 1.0) * err).mean()
        penalty = (F.relu(self.min_spread - (pred[:, 1] - pred[:, 0])).mean()
                   + F.relu(self.min_spread - (pred[:, 2] - pred[:, 1])).mean())
        return pinball + self.lam * penalty


class MonotoneHead(nn.Module):
    """Returns the 1%, 2.5% and 5% quantiles, ordered: -VaR_1 < -VaR_2.5 < -VaR_5."""

    def __init__(self, n_in, eps):
        super().__init__()
        self.eps, self.base_head, self.spread_head = eps, nn.Linear(n_in, 1), nn.Linear(n_in, 2)

    def forward(self, x):
        var5 = F.softplus(self.base_head(x))
        spreads = F.relu(self.spread_head(x)) + self.eps
        var25 = var5 + spreads[:, 0:1]
        return torch.cat([-(var25 + spreads[:, 1:2]), -var25, -var5], dim=1)


class FFNN(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        h, d = D["hidden_size"], D["dropout"]
        self.backbone = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(d), nn.Linear(h, h), nn.ReLU(), nn.Dropout(d),
                                      nn.Linear(h, h), nn.ReLU(), nn.Dropout(d))
        self.output_head = MonotoneHead(h, D["monotonic_epsilon"])

    def forward(self, x):
        return self.output_head(self.backbone(x))


class RNN(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        h, rh, d = D["hidden_size"], D["rnn_hidden"], D["dropout"]
        self.input_proj = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(d))
        self.rnn = nn.RNN(h, rh, batch_first=True)
        self.backbone_head = nn.Sequential(nn.Dropout(d), nn.Linear(rh, h), nn.ReLU(), nn.Dropout(d))
        self.output_head = MonotoneHead(h, D["monotonic_epsilon"])

    def forward(self, x):
        out, _ = self.rnn(self.input_proj(x).unsqueeze(1))
        return self.output_head(self.backbone_head(out[:, -1, :]))


class LSTM(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        h, rh, d = D["hidden_size"], D["rnn_hidden"], D["dropout"]
        self.input_proj = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(d))
        self.lstm1 = nn.LSTM(h, rh, batch_first=True)
        self.bottleneck = nn.Sequential(nn.Dropout(d), nn.Linear(rh, 1), nn.ReLU())
        self.lstm2 = nn.LSTM(1, rh, batch_first=True)
        self.backbone_head = nn.Sequential(nn.Dropout(d), nn.Linear(rh, h), nn.ReLU())
        self.output_head = MonotoneHead(h, D["monotonic_epsilon"])

    def forward(self, x):
        out1, _ = self.lstm1(self.input_proj(x).unsqueeze(1))
        out2, _ = self.lstm2(self.bottleneck(out1))
        return self.output_head(self.backbone_head(out2[:, -1, :]))


NETS = {"FFNN": FFNN, "RNN": RNN, "LSTM": LSTM}


def train(net, x_fit, y_fit, x_val, y_val):
    """Adam on mini-batches in time order; stop when the validation loss has not improved for `patience` epochs."""
    torch.manual_seed(D["seed"])
    model = net(x_fit.shape[1]).to(DEVICE)
    loss_fn = PinballLoss(CFG["quantiles"], D["spread_penalty_lambda"], D["spread_penalty_min"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=D["learning_rate"])
    tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    loader = DataLoader(TensorDataset(tensor(x_fit), tensor(y_fit.reshape(-1, 1))), batch_size=D["batch_size"], shuffle=False)
    x_va, y_va = tensor(x_val), tensor(y_val.reshape(-1, 1))
    best, best_state, wait = float("inf"), None, 0
    for _ in range(D["max_epochs"]):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_va), y_va).item()
        if val < best - 1e-7:
            best, wait, best_state = val, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= D["patience"]:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


def network(model: str, pair: str, run: dict):
    """The deep models always read ATM and RR25 of the run's tenor (plus the pair extras)."""
    df = master(pair, tenor=run["tenor"], features=["ATM", "RR25"], test_end=run["test_end"])
    train_df, test_df = split(df, run)
    if len(test_df) < 30:
        return None
    atm, rr = df["ATM"].values.astype(np.float32), df["RR25"].values.astype(np.float32)
    x = np.column_stack([atm, rr, atm * rr] + [df[c].values.astype(np.float32) for c in EXTRA.get(pair, []) if c in df])
    is_train = (df.index >= CFG["sample"]["train_start"]) & (df.index <= run["train_end"])
    is_test = (df.index >= run["test_start"]) & (df.index <= run["test_end"])
    mean, std = x[is_train].mean(axis=0), x[is_train].std(axis=0) + 1e-8
    x_train, x_test = (x[is_train] - mean) / std, (x[is_test] - mean) / std
    y = df["return_fwd"].values.astype(np.float32)
    n_fit = int(len(x_train) * D["train_split"])
    net = train(NETS[model], x_train[:n_fit], y[is_train][:n_fit], x_train[n_fit:], y[is_train][n_fit:])
    net.eval()
    with torch.no_grad():
        q_hat = net(torch.tensor(x_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    var = {q: -q_hat[:, i] for i, q in enumerate(CFG["quantiles"])}
    return forecasts(pair, model, df.index[is_test], y[is_test].astype(np.float64), var)
