"""Pure-NumPy sliding-window builder for sequence models (no torch dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.preprocessing.splits import SplitDates


def make_windows(df: pd.DataFrame, input_cols: list[str], target_col: str, input_len: int, horizon: int,
                 splits: SplitDates, observed_col: str | None = None,
                 require_observed: bool = True) -> dict[str, np.ndarray | pd.DataFrame]:
    """Build (X, Y) windows per sensor.

    X[i] = rows [i, i+input_len);  Y[i] = target_col of the next ``horizon`` rows.
    Windows containing NaN are dropped. If ``observed_col`` is given and ``require_observed`` is True (default), a window
    is also dropped unless EVERY input day and EVERY target day is a real observation (no interpolated value anywhere).
    Accounting is returned in ``stats`` (per split: candidate / retained / dropped_imputed). Each window gets a split label based on its target days
    (see ``splits`` module); straddling windows get label 'discard'.
    Returns dict with X (N, L, F), Y (N, H), obs (N, H bool), meta (DataFrame).
    """
    Xs, Ys, Os, metas = [], [], [], []
    counts: dict[str, dict[str, int]] = {k: {"candidate": 0, "retained": 0, "dropped_imputed": 0} for k in ("train", "val", "test", "discard")}
    for sid, g in df.groupby("sensor_id", sort=True):
        g = g.sort_values("date")
        T = len(g)
        if T < input_len + horizon:
            continue
        if not (g["date"].diff().dropna() == pd.Timedelta(days=1)).all():
            raise ValueError(f"{sid}: dates are not a contiguous daily calendar")
        arr = g[input_cols].to_numpy(dtype=float)
        tgt = g[target_col].to_numpy(dtype=float)
        obs = g[observed_col].to_numpy(dtype=float) if observed_col else np.ones(T)
        dates = g["date"].to_numpy()
        n = T - input_len - horizon + 1
        Xw = sliding_window_view(arr, input_len, axis=0).transpose(0, 2, 1)[:n]
        Yw = sliding_window_view(tgt, horizon)[input_len:input_len + n]
        Ow = sliding_window_view(obs, horizon)[input_len:input_len + n] == 1
        complete = ~np.isnan(Xw).any(axis=(1, 2)) & ~np.isnan(Yw).any(axis=1)
        if observed_col and require_observed:
            Iw = sliding_window_view(obs, input_len)[:n] == 1          # input-span observation flags
            real = Iw.all(axis=1) & Ow.all(axis=1)
        else:
            real = np.ones(n, bool)
        ok = complete & real
        idx = np.arange(n)
        first_t = pd.to_datetime(dates[idx + input_len])
        last_t = pd.to_datetime(dates[idx + input_len + horizon - 1])
        label = np.full(n, "discard", dtype=object)
        label[(first_t > splits.val_end)] = "test"
        label[(first_t > splits.train_end) & (last_t <= splits.val_end)] = "val"
        label[last_t <= splits.train_end] = "train"
        for k in counts:
            m = label == k
            counts[k]["candidate"] += int((m & complete).sum()); counts[k]["retained"] += int((m & ok).sum())
            counts[k]["dropped_imputed"] += int((m & complete & ~real).sum())
        metas.append(pd.DataFrame({"sensor_id": sid, "origin_date": pd.to_datetime(dates[idx + input_len - 1]),
                                   "split": label})[ok])
        Xs.append(Xw[ok]); Ys.append(Yw[ok]); Os.append(Ow[ok])
    if not Xs:
        raise ValueError("No windows could be built (not enough contiguous data)")
    return {"X": np.concatenate(Xs), "Y": np.concatenate(Ys), "obs": np.concatenate(Os),
            "meta": pd.concat(metas, ignore_index=True), "stats": counts}
