"""Re-train all models on the EXISTING data CSV (does not regenerate synthetic data).

    python scripts/train_all.py [--source synthetic|real]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--lstm-epochs", type=int, default=None)
    a = ap.parse_args()
    run(a.source, regenerate=False, skip_lstm=a.skip_lstm, lstm_epochs=a.lstm_epochs)


if __name__ == "__main__":
    main()
