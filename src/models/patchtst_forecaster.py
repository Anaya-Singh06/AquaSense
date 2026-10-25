"""PatchTST forecaster (PyTorch): patched, channel-independent Transformer for multi-day-ahead groundwater depth.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023).

What is PatchTST here (and what is adapted)
  * Each input channel (depth, log1p rain, temperature, ET, soil moisture, sin/cos day-of-year) is cut into overlapping
    PATCHES of ``patch_len`` days every ``stride`` days. Each patch is linearly embedded to ``d_model`` and given a learned
    positional embedding; a Transformer encoder (self-attention over the patch tokens) is applied to every channel
    independently with SHARED weights (channel independence, as in the paper).
  * The per-channel token sequences are flattened and concatenated (channel order is fixed) into one linear head that
    outputs the ``horizon`` future depths. (The paper forecasts every channel; AquaSense needs groundwater depth only, so the
    head is multi-channel-in / depth-out. ``use_covariates=false`` gives the pure univariate depth-only variant.)
  * Optional reversible instance normalisation (RevIN, on by default): each window's non-calendar channels are centred and
    scaled by their own mean/std before the encoder. The calendar channels (sin/cos) are NOT instance-normalised because that
    would erase the absolute season.
  * Like the project's LSTM the network starts from PERSISTENCE: prediction = last observed (scaled) depth + scale * head(...),
    where scale = window std of depth with RevIN and 1 without. The head is zero-initialised, so an untrained network
    reproduces persistence exactly.
  * Deviations from the paper, deliberately kept simple: pre-LayerNorm encoder layers (instead of BatchNorm), the oldest
    ``(context_length - patch_len) % stride`` days are dropped so patches end on the most recent day (instead of padding),
    and no self-supervised pre-training.

Everything that is not the network is reused from the project: ``LSTMForecaster`` scaling helpers (scalers fitted on TRAIN rows only),
``make_windows`` (per-station windows, contiguous daily calendar, every input and target day a REAL observation, split by
target dates), ``evaluate_against_persistence`` metrics.

Artifacts (never the LSTM / XGBoost / LightGBM names), in the source's models directory, for horizon h:
  ``patchtst_{h}d.pt`` (weights + config), ``patchtst_{h}d_scalers.joblib``, ``patchtst_{h}d_meta.json`` (preprocessing metadata,
  split dates, window accounting, training history), and ``patchtst_{h}d_resume.pt`` (only while training; removed when it finishes).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import REPO_ROOT, SEED, TARGET, get_logger
from src.evaluation.metrics import evaluate_against_persistence, persistence_forecast
from src.models.lstm_forecaster import SCALED_COLS, LSTMForecaster
from src.preprocessing.splits import SplitDates
from src.preprocessing.windows import make_windows

log = get_logger("patchtst")
CONFIG_PATH = REPO_ROOT / "config" / "patchtst_config.json"
FORMAT_VERSION = 1
DEPTH_COL = "x_depth"            # must be the FIRST model channel (persistence anchor)


# ------------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class PatchTSTConfig:
    # model
    context_length: int = 96
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.2
    head_dropout: float = 0.1
    revin: bool = True
    use_covariates: bool = True
    # training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    eval_batch_size: int = 512
    max_epochs: int = 60
    patience: int = 8
    min_delta: float = 1e-6
    grad_clip: float = 1.0
    amp: str = "auto"                # "auto" (CUDA only) | "on" (CUDA only) | "off"
    num_workers: int = 0             # 0 is the safe choice on Windows
    seed: int = SEED
    # inference
    strict_inference_inputs: bool = True   # skip stations whose last context contains interpolated depth values

    def validate(self) -> "PatchTSTConfig":
        if self.patch_len < 1 or self.stride < 1:
            raise ValueError("patch_len and stride must be >= 1")
        if self.context_length < self.patch_len:
            raise ValueError(f"context_length ({self.context_length}) must be >= patch_len ({self.patch_len})")
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_layers < 1 or self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("n_layers, batch_size, max_epochs and patience must be >= 1")
        if not (0.0 <= self.dropout < 1.0 and 0.0 <= self.head_dropout < 1.0):
            raise ValueError("dropout and head_dropout must be in [0, 1)")
        if self.amp not in ("auto", "on", "off"):
            raise ValueError("amp must be 'auto', 'on' or 'off'")
        return self

    @property
    def n_patches(self) -> int:
        return (self.context_length - self.patch_len) // self.stride + 1


def load_patchtst_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read the settings file -> {'config': PatchTSTConfig, 'horizons': [int], 'rolling_max_epochs': int}."""
    raw = json.loads(Path(path or CONFIG_PATH).read_text())
    flat: dict[str, Any] = {}
    for section in ("model", "training", "inference"):
        flat.update(raw.get(section, {}))
    if isinstance(flat.get("amp"), bool):
        flat["amp"] = "on" if flat["amp"] else "off"
    unknown = set(flat) - {f.name for f in fields(PatchTSTConfig)}
    if unknown:
        raise ValueError(f"unknown PatchTST setting(s): {sorted(unknown)}")
    cfg = PatchTSTConfig(**flat).validate()
    horizons = [int(h) for h in raw.get("horizons", [7])]
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("'horizons' must be a non-empty list of positive integers")
    return {"config": cfg, "horizons": horizons,
            "rolling_max_epochs": int(raw.get("rolling_origin", {}).get("max_epochs", cfg.max_epochs))}


def artifact_paths(models_dir: Path, horizon: int) -> dict[str, Path]:
    """All PatchTST file names for one horizon (all start with ``patchtst_``)."""
    stem = f"patchtst_{horizon}d"
    return {"model": models_dir / f"{stem}.pt", "scalers": models_dir / f"{stem}_scalers.joblib",
            "meta": models_dir / f"{stem}_meta.json", "resume": models_dir / f"{stem}_resume.pt"}


# ------------------------------------------------------------------------------------------------ network
class PatchTSTNet(nn.Module):
    """Input (B, context_length, C) in model units -> (B, horizon) depths in model units (channel 0 = depth)."""

    def __init__(self, n_channels: int, context_length: int, horizon: int, patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3, d_ff: int = 128, dropout: float = 0.2,
                 head_dropout: float = 0.1, revin: bool = True, norm_mask: list[bool] | None = None) -> None:
        super().__init__()
        if context_length < patch_len:
            raise ValueError("context_length must be >= patch_len")
        self.n_channels, self.context_length, self.horizon = n_channels, context_length, horizon
        self.patch_len, self.stride, self.revin = patch_len, stride, revin
        self.n_patches = (context_length - patch_len) // stride + 1
        self.trim = (context_length - patch_len) % stride       # oldest days dropped so patches end at the latest day
        mask = torch.tensor(norm_mask if norm_mask is not None else [True] * n_channels, dtype=torch.bool)
        if mask.numel() != n_channels:
            raise ValueError("norm_mask must have one entry per channel")
        self.register_buffer("norm_mask", mask.view(1, 1, -1), persistent=False)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.embed_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(n_channels * self.n_patches * d_model, horizon)
        nn.init.zeros_(self.head.weight)       # untrained network == persistence
        nn.init.zeros_(self.head.bias)

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        """(B, L, C) -> (B*C, n_patches, patch_len): every channel becomes its own univariate sequence of patches."""
        b, _, c = z.shape
        z = z[:, self.trim:, :].permute(0, 2, 1).reshape(b * c, -1)
        return z.unfold(-1, self.patch_len, self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, c = x.shape
        if length != self.context_length or c != self.n_channels:
            raise ValueError(f"expected input (B, {self.context_length}, {self.n_channels}), got {tuple(x.shape)}")
        last = x[:, -1, 0]                                           # last observed depth (persistence anchor)
        if self.revin:
            mean = x.mean(dim=1, keepdim=True)
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            z = torch.where(self.norm_mask, (x - mean) / std, x)
            scale = std[:, 0, 0]
        else:
            z, scale = x, torch.ones_like(last)
        h = self.embed_drop(self.patch_embed(self.patchify(z)) + self.pos_embed)   # (B*C, N, D)
        h = self.encoder(h)
        delta = self.head(self.head_drop(h.reshape(b, -1)))          # (B, horizon)
        return last.unsqueeze(1) + scale.unsqueeze(1) * delta.float()


class _ArrayDS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X, self.Y = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.Y[i]


def _grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):          # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ------------------------------------------------------------------------------------------------ forecaster
class PatchTSTForecaster(LSTMForecaster):
    """Training / evaluation / inference wrapper. Inherits ONLY the LSTM scaling helpers (``_fit_scalers``, ``_scale``,
    ``_inv_depth``, ``_log_device``); everything else is overridden, so LSTM behaviour is untouched."""

    def __init__(self, horizon: int = 7, config: PatchTSTConfig | None = None, device: str = "auto") -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = int(horizon)
        self.config = (config or PatchTSTConfig()).validate()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.cols = list(SCALED_COLS) if self.config.use_covariates else [DEPTH_COL]
        self.net: PatchTSTNet | None = None
        self.scalers: dict[str, Any] = {}
        self.history: list[dict[str, float]] = []
        self.window_stats: dict[str, dict[str, int]] = {}
        self.device_info: dict[str, Any] = {}
        self.split_dates: dict[str, str] = {}
        self.best_epoch: int | None = None
        self.best_val_loss: float | None = None
        self.persistence_val_loss: float | None = None   # scaled-depth MSE of plain persistence on the validation windows
        self.peak_gpu_mb: float | None = None

    # ---- construction helpers ------------------------------------------------------
    def _build_net(self) -> PatchTSTNet:
        c = self.config
        return PatchTSTNet(len(self.cols), c.context_length, self.horizon, c.patch_len, c.stride, c.d_model, c.n_heads, c.n_layers,
                           c.d_ff, c.dropout, c.head_dropout, c.revin, norm_mask=[col.startswith("x_") for col in self.cols])

    def _use_amp(self) -> bool:
        if self.config.amp == "off":
            return False
        if self.device.type != "cuda":
            if self.config.amp == "on":
                log.warning("amp='on' ignored: mixed precision is only used on CUDA.")
            return False
        return True

    # ---- training ---------------------------------------------------------------------
    def fit(self, clean: pd.DataFrame, splits: SplitDates, checkpoint_dir: Path | None = None, resume: bool = False) -> dict[str, Any]:
        """Train on the 'train' windows, early-stop on 'val' (MSE in scaled-depth units). Returns the windows dict.

        ``checkpoint_dir``: write ``patchtst_{h}d_resume.pt`` after every epoch; ``resume=True`` continues from it if present."""
        cfg = self.config
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self._log_device()
        self.split_dates = splits.as_dict()
        self._fit_scalers(clean, splits.train_end)                       # TRAIN rows only (observed depth, date <= train_end)
        scaled = self._scale(clean)
        scaled["depth_observed"] = (scaled["depth_imputed"] == 0).astype(int) if "depth_imputed" in scaled else 1
        w = make_windows(scaled, self.cols, DEPTH_COL, cfg.context_length, self.horizon, splits, observed_col="depth_observed")
        w["X"], w["Y"] = w["X"].astype(np.float32), w["Y"].astype(np.float32)
        self.window_stats = w["stats"]
        sp = w["meta"]["split"].to_numpy()
        tr, va = sp == "train", sp == "val"
        if tr.sum() < 100 or va.sum() < 10:
            raise ValueError(f"Too few PatchTST windows: train={int(tr.sum())} val={int(va.sum())} "
                             f"(context_length={cfg.context_length}, horizon={self.horizon}; all input and target days must be observed)")
        gen = torch.Generator().manual_seed(cfg.seed)
        pin = self.device.type == "cuda"
        dl_tr = DataLoader(_ArrayDS(w["X"][tr], w["Y"][tr]), batch_size=cfg.batch_size, shuffle=True, generator=gen,
                           num_workers=cfg.num_workers, pin_memory=pin)
        dl_va = DataLoader(_ArrayDS(w["X"][va], w["Y"][va]), batch_size=cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=pin)
        n_tr, n_va = int(tr.sum()), int(va.sum())
        # Reference only (no effect on training or checkpoint selection): the head is zero-initialised, so the UNTRAINED network is
        # persistence. Its validation MSE (same units as the logged val loss) shows whether the selected epoch beats its own start.
        self.persistence_val_loss = float(np.mean((w["X"][va][:, -1, :1].astype(np.float64) - w["Y"][va].astype(np.float64)) ** 2))

        self.net = self._build_net().to(self.device)
        log.info("PatchTST %dd: %d parameters | context=%d patch=%d stride=%d -> %d patches/channel x %d channels | train=%d val=%d windows",
                 self.horizon, sum(p.numel() for p in self.net.parameters()), cfg.context_length, cfg.patch_len, cfg.stride,
                 self.net.n_patches, len(self.cols), n_tr, n_va)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        use_amp = self._use_amp()
        scaler = _grad_scaler(use_amp)
        loss_fn = nn.MSELoss()
        start, best, best_state, bad, best_ep = 0, float("inf"), None, 0, 0
        self.history = []
        resume_path = artifact_paths(checkpoint_dir, self.horizon)["resume"] if checkpoint_dir is not None else None
        if resume and resume_path is not None and resume_path.exists():
            ck = torch.load(resume_path, map_location="cpu", weights_only=False)   # local file written by this class
            self._check_resume(ck, splits)
            self.net.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            start, best, best_state, bad, best_ep, self.history = ck["epoch"], ck["best"], ck["best_state"], ck["bad"], ck["best_ep"], ck["history"]
            log.info("resumed from %s at epoch %d (best val loss %.5f)", resume_path.name, start, best)
        elif resume:
            log.info("--resume given but no resume checkpoint found; training from scratch")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for ep in range(start + 1, cfg.max_epochs + 1):
            if bad >= cfg.patience:
                break
            t0 = time.time()
            gen.manual_seed(cfg.seed + ep)                                # reproducible shuffling, also after a resume
            self.net.train(); tl = 0.0
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    pred = self.net(xb)
                loss = loss_fn(pred.float(), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
                tl += loss.item() * len(xb)
            self.net.eval(); vl = 0.0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                        pred = self.net(xb)
                    vl += loss_fn(pred.float(), yb).item() * len(xb)
            tl, vl = tl / n_tr, vl / n_va
            if not math.isfinite(vl) or not math.isfinite(tl):
                raise RuntimeError(f"non-finite loss at epoch {ep} (train={tl}, val={vl}); try amp='off' or a lower lr")
            self.history.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "seconds": round(time.time() - t0, 2)})
            if vl < best - cfg.min_delta:
                best, bad, best_ep = vl, 0, ep
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
            log.info("epoch %3d  train %.5f  val %.5f  best %.5f  (%.1fs)%s", ep, tl, vl, best, time.time() - t0,
                     "" if bad == 0 else f"  no-improve {bad}/{cfg.patience}")
            if resume_path is not None:
                self._write_resume(resume_path, ep, opt, scaler, best, best_state, bad, best_ep, splits)
        if best_state is None:
            raise RuntimeError("training produced no valid state (max_epochs reached before any epoch ran?)")
        if bad >= cfg.patience:
            log.info("early stop (best val loss %.5f)", best)
        self.net.load_state_dict(best_state)
        self.best_val_loss = float(best)
        self.best_epoch = int(best_ep)
        if self.best_val_loss >= self.persistence_val_loss:
            log.warning("PatchTST %dd: best validation loss %.5f (epoch %d) is NOT better than the persistence start point %.5f - the "
                        "model has not yet learned anything useful on validation (too few epochs/steps, or no signal beyond persistence).",
                        self.horizon, self.best_val_loss, self.best_epoch, self.persistence_val_loss)
        else:
            log.info("PatchTST %dd: best validation loss %.5f vs persistence %.5f", self.horizon, self.best_val_loss, self.persistence_val_loss)
        if self.device.type == "cuda":
            self.peak_gpu_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            log.info("peak GPU memory allocated: %.1f MiB", self.peak_gpu_mb)
        if resume_path is not None and resume_path.exists():
            resume_path.unlink()                                           # finished: the final artifacts replace it
        return w

    def _write_resume(self, path: Path, epoch: int, opt: Any, scaler: Any, best: float, best_state: Any, bad: int, best_ep: int,
                      splits: SplitDates) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"epoch": epoch, "state_dict": self.net.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best": float(best), "best_state": best_state, "bad": int(bad), "best_ep": int(best_ep), "history": self.history,
                    "config": asdict(self.config), "horizon": self.horizon, "cols": self.cols, "splits": splits.as_dict()}, tmp)
        tmp.replace(path)

    def _check_resume(self, ck: dict[str, Any], splits: SplitDates) -> None:
        keep = lambda d: {k: v for k, v in d.items() if k not in ("max_epochs", "patience", "num_workers")}  # noqa: E731
        if (ck["horizon"] != self.horizon or ck["cols"] != self.cols or ck["splits"] != splits.as_dict()
                or keep(ck["config"]) != keep(asdict(self.config))):
            raise ValueError("resume checkpoint was made with different settings / data splits; delete it or drop --resume")

    # ---- evaluation -----------------------------------------------------------------------
    @torch.no_grad()
    def _predict_scaled(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        bs = self.config.eval_batch_size
        out = [self.net(torch.from_numpy(np.ascontiguousarray(X[i:i + bs], dtype=np.float32)).to(self.device)).cpu().numpy()
               for i in range(0, len(X), bs)]
        return np.concatenate(out) if out else np.empty((0, self.horizon), np.float32)

    def evaluate(self, w: dict[str, Any], split: str = "test") -> dict[str, Any]:
        """Metrics vs persistence on windows of one split (all input and target days real observations).
        Same structure as ``LSTMForecaster.evaluate``; ``day{h}_only`` scores the h-day-ahead end point alone."""
        m = (w["meta"]["split"].to_numpy() == split) & w["obs"].all(axis=1)
        if m.sum() == 0:
            raise ValueError(f"no windows for split {split}")
        h = self.horizon
        pred = self._inv_depth(self._predict_scaled(w["X"][m]))
        true = self._inv_depth(w["Y"][m])
        last = self._inv_depth(w["X"][m][:, -1, 0])
        base = persistence_forecast(last, h)
        sens = w["meta"]["sensor_id"].to_numpy()[m]
        res = evaluate_against_persistence(true, pred, base, np.repeat(sens, h))
        res[f"day{h}_only"] = evaluate_against_persistence(true[:, -1], pred[:, -1], base[:, -1], sens)
        res.update(horizon_days=h, n_windows=int(m.sum()), split=split)
        res["_arrays"] = {"true": true, "pred": pred, "base": base, "meta": w["meta"][m].reset_index(drop=True)}
        return res

    # ---- inference ----------------------------------------------------------------------
    def forecast_latest(self, clean: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        """``horizon``-day forecast per sensor from its last ``context_length`` contiguous days (same output shape as the LSTM).

        A sensor is skipped (with a warning) if its context is incomplete, contains NaN, or - when
        ``strict_inference_inputs`` (default) - contains interpolated depth values, because training windows never contain them."""
        scaled = self._scale(clean)
        L = self.config.context_length
        sids, arrays, last_dates = [], [], []
        for sid, g in scaled.groupby("sensor_id", sort=True):
            g = g.sort_values("date").tail(L)
            x = g[self.cols].to_numpy(dtype=float)
            if len(g) < L or np.isnan(x).any() or not (g["date"].diff().dropna() == pd.Timedelta(days=1)).all():
                log.warning("PatchTST: skipping %s (incomplete last %d days)", sid, L)
                continue
            if self.config.strict_inference_inputs and "depth_imputed" in g and (g["depth_imputed"] > 0).any():
                log.warning("PatchTST: skipping %s (interpolated depth values inside the last %d days)", sid, L)
                continue
            sids.append(sid); arrays.append(x); last_dates.append(g["date"].iloc[-1])
        if not sids:
            log.warning("PatchTST: no sensor has a usable last %d-day context (strict_inference_inputs=%s) -> no forecasts. With a long "
                        "context and gappy data this is expected: every sensor has an interpolated/missing day in its last %d days.",
                        L, self.config.strict_inference_inputs, L)
            return {}
        pred = self._inv_depth(self._predict_scaled(np.stack(arrays)))
        return {sid: [{"date": str((d0 + pd.Timedelta(days=k + 1)).date()), "depth_m_bgl": round(float(v), 3)} for k, v in enumerate(row)]
                for sid, d0, row in zip(sids, last_dates, pred)}

    # ---- persistence ----------------------------------------------------------------------
    def save(self, models_dir: Path) -> Path:
        """Write ``patchtst_{h}d.pt`` / ``_scalers.joblib`` / ``_meta.json`` into ``models_dir`` (never other models' names)."""
        models_dir.mkdir(parents=True, exist_ok=True)
        p = artifact_paths(models_dir, self.horizon)
        torch.save({"format_version": FORMAT_VERSION, "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                    "config": asdict(self.config), "horizon": self.horizon, "cols": self.cols}, p["model"])
        joblib.dump(self.scalers, p["scalers"])
        meta = {"model": "PatchTST", "format_version": FORMAT_VERSION, "horizon_days": self.horizon, "config": asdict(self.config),
                "input_columns": self.cols, "target": TARGET, "units": "m below ground level (higher = deeper)",
                "split_dates": self.split_dates, "n_parameters": int(sum(p_.numel() for p_ in self.net.parameters())),
                "scalers_fit_on": "TRAIN rows only (date <= train_end, observed depth, complete climate inputs)",
                "scaler_params": {k: {"mean": float(s.mean_[0]), "scale": float(s.scale_[0])} for k, s in self.scalers.items()},
                "window_accounting": self.window_stats, "best_epoch": self.best_epoch, "best_val_loss": self.best_val_loss,
                "persistence_val_loss": self.persistence_val_loss,
                "best_val_beats_persistence": (None if self.best_val_loss is None or self.persistence_val_loss is None
                                               else bool(self.best_val_loss < self.persistence_val_loss)),
                "epochs_run": len(self.history), "history": self.history, "device": self.device_info, "peak_gpu_mb": self.peak_gpu_mb,
                "leakage_rules": "every input day and every target day of a window is a real (non-interpolated) observation; windows never "
                                 "cross stations; windows are assigned to splits by their target dates; straddling windows are discarded"}
        p["meta"].write_text(json.dumps(meta, indent=2, default=str))
        return p["model"]

    @classmethod
    def load(cls, models_dir: Path, horizon: int, device: str = "auto") -> "PatchTSTForecaster":
        p = artifact_paths(models_dir, horizon)
        if not p["model"].exists() or not p["scalers"].exists():
            raise FileNotFoundError(f"{p['model']} / {p['scalers']} not found - run scripts/train_patchtst.py first")
        ck = torch.load(p["model"], map_location="cpu", weights_only=True)
        if ck.get("format_version") != FORMAT_VERSION or ck["horizon"] != horizon:
            raise ValueError(f"{p['model'].name}: unsupported format / horizon mismatch - retrain")
        obj = cls(horizon, PatchTSTConfig(**ck["config"]), device)
        if ck["cols"] != obj.cols:
            raise ValueError("saved PatchTST input columns do not match the configuration - retrain")
        obj.net = obj._build_net().to(obj.device)
        obj.net.load_state_dict(ck["state_dict"])
        obj.net.eval()
        obj.scalers = joblib.load(p["scalers"])
        return obj


def replace_config(cfg: PatchTSTConfig, **overrides: Any) -> PatchTSTConfig:
    """Copy of ``cfg`` with some fields replaced (e.g. ``max_epochs`` for a smoke run), re-validated."""
    return replace(cfg, **overrides).validate()
