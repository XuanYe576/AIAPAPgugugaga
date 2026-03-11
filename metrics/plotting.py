"""Shared matplotlib loader for metrics and plotting helpers."""

from __future__ import annotations


def load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to create figures. "
            "Install it with: pip install matplotlib"
        ) from exc
    return plt
