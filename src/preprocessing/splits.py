"""Chronological train/validation/test splitting (never shuffled).

Rule: a sample belongs to the split containing its **target date(s)**.
  train: target_date <= train_end;  val: train_end < target_date <= val_end;  test: target_date > val_end.
For multi-day targets (LSTM) a window must lie completely inside one split; windows straddling a
boundary are discarded. Inputs may reach back into earlier periods (they are the past).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitDates:
    start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    end: pd.Timestamp

    def as_dict(self) -> dict[str, str]:
        return {k: str(v.date()) for k, v in self.__dict__.items()}


def make_split_dates(dates: pd.Series | pd.DatetimeIndex, train_frac: float = 0.70, val_frac: float = 0.15) -> SplitDates:
    """Boundary dates splitting the unique calendar into 70% / 15% / 15% (by number of days)."""
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("fractions must satisfy 0 < train, val and train + val < 1")
    u = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).unique())).sort_values()
    n = len(u)
    if n < 20:
        raise ValueError("Not enough unique dates to split")
    return SplitDates(u[0], u[int(n * train_frac) - 1], u[int(n * (train_frac + val_frac)) - 1], u[-1])


def assign_split_by_target(target_date: pd.Series, s: SplitDates) -> pd.Series:
    """Label each target date 'train' / 'val' / 'test'."""
    out = pd.Series("test", index=target_date.index, dtype=object)
    out[target_date <= s.val_end] = "val"
    out[target_date <= s.train_end] = "train"
    return out
