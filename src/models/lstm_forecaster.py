"""PyTorch LSTM: 30 days of history (depth + climate + season) -> next 7 days of depth.

Design notes
  * Inputs per day: depth, log1p(rainfall), temperature, ET, soil moisture (all standardised with
    scalers fitted on TRAIN rows only) + sin/cos day-of-year.
  * The network outputs 7 *changes* relative to the last observed (scaled) depth, i.e. it starts from the
    persistence forecast and learns a correction. Persistence is therefore the natural benchmark.
  * Chronological windows (see ``src.preprocessing.splits``); early stopping on validation MSE.
Artifacts: ``models/lstm_model.pt`` (weights + config) and ``models/lstm_scalers.joblib``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import SEED, TARGET, get_logger
from src.evaluation.metrics import evaluate_against_persistence, persistence_forecast
from src.preprocessing.splits import SplitDates
from src.preprocessing.windows import make_windows

log = get_logger("lstm")
INPUT_LEN, HORIZON = 30, 7
SCALED_COLS = ["x_depth", "x_rain", "x_temp", "x_et", "x_sm", "sin_doy", "cos_doy"]


class _WindowDS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X, self.Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.Y[i]


class LSTMNet(nn.Module):
    """LSTM encoder + linear head predicting 7 deltas added to the last scaled depth (feature 0)."""

    def __init__(self, n_features: int = len(SCALED_COLS), hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, horizon: int = HORIZON) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return x[:, -1, 0:1] + self.head(out[:, -1])


class LSTMForecaster:
    """Training / evaluation / inference wrapper."""

    def __init__(self, hidden: int = 64, layers: int = 2, lr: float = 1e-3, batch_size: int = 128,
                 max_epochs: int = 60, patience: int = 8, seed: int = SEED) -> None:
        self.cfg = dict(hidden=hidden, layers=layers, lr=lr, batch_size=batch_size, max_epochs=max_epochs,
                        patience=patience, seed=seed, input_len=INPUT_LEN, horizon=HORIZON)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: LSTMNet | None = None
        self.scalers: dict[str, StandardScaler] = {}
        self.history: list[dict[str, float]] = []
        self.window_stats: dict[str, dict[str, int]] = {}
        self.device_info: dict[str, Any] = {}

    def _log_device(self) -> None:
        import platform
        cuda = torch.cuda.is_available()
        self.device_info = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                            "cuda_available": cuda, "device": str(self.device),
                            "gpu": torch.cuda.get_device_name(0) if cuda else None}
        log.info("Python %s | torch %s | torch CUDA %s | cuda_available=%s", self.device_info["python"], torch.__version__,
                 torch.version.cuda, cuda)
        log.info("Device: %s%s", self.device, f" | GPU: {self.device_info['gpu']}" if cuda else "")
        if not cuda:
            log.warning("WARNING: CUDA unavailable; training on CPU.")

    # ---- scaling -------------------------------------------------------------
    def _fit_scalers(self, df: pd.DataFrame, train_end: pd.Timestamp) -> None:
        real = df["depth_imputed"] == 0 if "depth_imputed" in df else pd.Series(True, index=df.index)
        tr = df[(df["date"] <= train_end) & real].dropna(subset=[TARGET, "rainfall_mm", "temperature_c", "evapotranspiration_mm", "soil_moisture"])
        self.scalers = {
            "depth": StandardScaler().fit(tr[[TARGET]]),
            "rain": StandardScaler().fit(np.log1p(tr[["rainfall_mm"]])),
            "temp": StandardScaler().fit(tr[["temperature_c"]]),
            "et": StandardScaler().fit(tr[["evapotranspiration_mm"]]),
            "sm": StandardScaler().fit(tr[["soil_moisture"]]),
        }

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.sort_values(["sensor_id", "date"]).copy()
        d["x_depth"] = self.scalers["depth"].transform(d[[TARGET]]).ravel()
        d["x_rain"] = self.scalers["rain"].transform(np.log1p(d[["rainfall_mm"]])).ravel()
        d["x_temp"] = self.scalers["temp"].transform(d[["temperature_c"]]).ravel()
        d["x_et"] = self.scalers["et"].transform(d[["evapotranspiration_mm"]]).ravel()
        d["x_sm"] = self.scalers["sm"].transform(d[["soil_moisture"]]).ravel()
        ang = 2 * np.pi * d["date"].dt.dayofyear / 365.25
        d["sin_doy"], d["cos_doy"] = np.sin(ang), np.cos(ang)
        return d

    def _inv_depth(self, x: np.ndarray) -> np.ndarray:
        return self.scalers["depth"].inverse_transform(np.asarray(x).reshape(-1, 1)).reshape(np.shape(x))

    # ---- training --------------------------------------------------------------
    def fit(self, clean: pd.DataFrame, splits: SplitDates, checkpoint: Path | None = None) -> dict[str, Any]:
        """Train on the 'train' windows, early-stop on 'val'. Returns the windows dict for evaluation."""
        torch.manual_seed(self.cfg["seed"]); np.random.seed(self.cfg["seed"])
        self._log_device()
        self._fit_scalers(clean, splits.train_end)
        scaled = self._scale(clean)
        scaled["depth_observed"] = (scaled["depth_imputed"] == 0).astype(int) if "depth_imputed" in scaled else 1
        w = make_windows(scaled, SCALED_COLS, "x_depth", INPUT_LEN, HORIZON, splits, observed_col="depth_observed")
        self.window_stats = w["stats"]
        sp = w["meta"]["split"].to_numpy()
        tr, va = sp == "train", sp == "val"
        if tr.sum() < 100 or va.sum() < 10:
            raise ValueError(f"Too few LSTM windows: train={tr.sum()} val={va.sum()}")
        g = torch.Generator().manual_seed(self.cfg["seed"])
        dl_tr = DataLoader(_WindowDS(w["X"][tr], w["Y"][tr]), batch_size=self.cfg["batch_size"], shuffle=True, generator=g)
        dl_va = DataLoader(_WindowDS(w["X"][va], w["Y"][va]), batch_size=512)
        self.net = LSTMNet(hidden=self.cfg["hidden"], layers=self.cfg["layers"]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        loss_fn = nn.MSELoss()
        best, best_state, bad = float("inf"), None, 0
        for ep in range(1, self.cfg["max_epochs"] + 1):
            self.net.train(); tl = 0.0
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(); loss = loss_fn(self.net(xb), yb); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
                tl += loss.item() * len(xb)
            self.net.eval(); vl = 0.0
            with torch.no_grad():
                for xb, yb in dl_va:
                    vl += loss_fn(self.net(xb.to(self.device)), yb.to(self.device)).item() * len(xb)
            tl, vl = tl / tr.sum(), vl / va.sum()
            self.history.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    log.info("early stop at epoch %d (best val loss %.5f)", ep, best)
                    break
        self.net.load_state_dict(best_state)
        if checkpoint is not None:
            self.save(checkpoint)
        return w

    # ---- evaluation ------------------------------------------------------------
    @torch.no_grad()
    def _predict_scaled(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = [self.net(torch.tensor(X[i:i + 1024], dtype=torch.float32, device=self.device)).cpu().numpy()
               for i in range(0, len(X), 1024)]
        return np.concatenate(out)

    def evaluate(self, w: dict[str, Any], split: str = "test") -> dict[str, Any]:
        """Metrics vs persistence on windows of one split (only fully observed target windows)."""
        m = (w["meta"]["split"].to_numpy() == split) & w["obs"].all(axis=1)
        if m.sum() == 0:
            raise ValueError(f"no windows for split {split}")
        pred = self._inv_depth(self._predict_scaled(w["X"][m]))
        true = self._inv_depth(w["Y"][m])
        last = self._inv_depth(w["X"][m][:, -1, 0])
        base = persistence_forecast(last, HORIZON)
        sens = np.repeat(w["meta"]["sensor_id"].to_numpy()[m], HORIZON)
        res = evaluate_against_persistence(true, pred, base, sens)
        res["day7_only"] = evaluate_against_persistence(true[:, -1], pred[:, -1], base[:, -1], w["meta"]["sensor_id"].to_numpy()[m])
        res.update(horizon_days=HORIZON, n_windows=int(m.sum()), split=split)
        res["_arrays"] = {"true": true, "pred": pred, "base": base, "meta": w["meta"][m].reset_index(drop=True)}
        return res

    # ---- inference -------------------------------------------------------------
    def forecast_latest(self, clean: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        """7-day forecast per sensor from the last 30 contiguous days. Sensors with NaN inputs are skipped."""
        scaled = self._scale(clean)
        out: dict[str, list[dict[str, Any]]] = {}
        for sid, g in scaled.groupby("sensor_id", sort=True):
            g = g.sort_values("date").tail(INPUT_LEN)
            if len(g) < INPUT_LEN or g[SCALED_COLS].isna().any(axis=None):
                log.warning("LSTM: skipping %s (incomplete last %d days)", sid, INPUT_LEN)
                continue
            pred = self._inv_depth(self._predict_scaled(g[SCALED_COLS].to_numpy()[None]))[0]
            d0 = g["date"].iloc[-1]
            out[sid] = [{"date": str((d0 + pd.Timedelta(days=k + 1)).date()), "depth_m_bgl": round(float(v), 3)}
                        for k, v in enumerate(pred)]
        return out

    # ---- persistence -----------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "config": self.cfg, "scaled_cols": SCALED_COLS}, path)
        joblib.dump(self.scalers, path.with_name("lstm_scalers.joblib"))

    @classmethod
    def load(cls, path: Path) -> "LSTMForecaster":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/run_pipeline.py first")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        c = ck["config"]
        obj = cls(hidden=c["hidden"], layers=c["layers"], lr=c["lr"], batch_size=c["batch_size"],
                  max_epochs=c["max_epochs"], patience=c["patience"], seed=c["seed"])
        obj.net = LSTMNet(hidden=c["hidden"], layers=c["layers"]).to(obj.device)
        obj.net.load_state_dict(ck["state_dict"])
        obj.scalers = joblib.load(path.with_name("lstm_scalers.joblib"))
        return obj
