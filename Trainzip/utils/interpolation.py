"""Curve interpolation helpers for S11 magnitude datasets."""

from __future__ import annotations

import numpy as np


def uniform_freq_axis(start: float, stop: float, points: int) -> np.ndarray:
    if points <= 1:
        raise ValueError("points must be greater than 1")
    return np.linspace(float(start), float(stop), int(points), dtype=np.float32)


def pchip_interpolate_rows(
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Shape-preserving cubic Hermite interpolation for row-wise curves.

    HFSS interpolating sweeps operate on solver basis responses, not on exported
    dB-only samples. For magnitude-only CSVs, PCHIP is the safer surrogate:
    it keeps sampled notch depths fixed and avoids cubic/rational overshoot.
    """

    x_old = np.asarray(x_old, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)
    y_old = np.asarray(y_old, dtype=np.float64)
    squeeze = False
    if y_old.ndim == 1:
        y_old = y_old[None, :]
        squeeze = True
    if y_old.ndim != 2:
        raise ValueError("y_old must be a 1D or 2D array")
    if x_old.ndim != 1 or x_new.ndim != 1:
        raise ValueError("x_old and x_new must be 1D arrays")
    if y_old.shape[1] != x_old.size:
        raise ValueError("y_old column count must match x_old length")
    if x_old.size < 2:
        raise ValueError("At least two source points are required")
    if not np.all(np.diff(x_old) > 0):
        raise ValueError("x_old must be strictly increasing")

    h = np.diff(x_old)
    delta = np.diff(y_old, axis=1) / h[None, :]
    slopes = np.zeros_like(y_old)

    if x_old.size == 2:
        slopes[:, 0] = delta[:, 0]
        slopes[:, 1] = delta[:, 0]
    else:
        d0 = ((2.0 * h[0] + h[1]) * delta[:, 0] - h[0] * delta[:, 1]) / (h[0] + h[1])
        mask = np.sign(d0) != np.sign(delta[:, 0])
        d0[mask] = 0.0
        mask = (np.sign(delta[:, 0]) != np.sign(delta[:, 1])) & (np.abs(d0) > 3.0 * np.abs(delta[:, 0]))
        d0[mask] = 3.0 * delta[:, 0][mask]
        slopes[:, 0] = d0

        dn = ((2.0 * h[-1] + h[-2]) * delta[:, -1] - h[-1] * delta[:, -2]) / (h[-1] + h[-2])
        mask = np.sign(dn) != np.sign(delta[:, -1])
        dn[mask] = 0.0
        mask = (np.sign(delta[:, -1]) != np.sign(delta[:, -2])) & (np.abs(dn) > 3.0 * np.abs(delta[:, -1]))
        dn[mask] = 3.0 * delta[:, -1][mask]
        slopes[:, -1] = dn

        for i in range(1, x_old.size - 1):
            prev_delta = delta[:, i - 1]
            next_delta = delta[:, i]
            same_sign = (prev_delta * next_delta) > 0.0
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            denom = (w1 / prev_delta[same_sign]) + (w2 / next_delta[same_sign])
            slopes[same_sign, i] = (w1 + w2) / denom

    interval = np.searchsorted(x_old, x_new, side="right") - 1
    interval = np.clip(interval, 0, x_old.size - 2)
    x0 = x_old[interval]
    step = h[interval]
    t = (x_new - x0) / step
    t2 = t * t
    t3 = t2 * t

    y0 = y_old[:, interval]
    y1 = y_old[:, interval + 1]
    m0 = slopes[:, interval]
    m1 = slopes[:, interval + 1]

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    out = h00[None, :] * y0 + h10[None, :] * step[None, :] * m0 + h01[None, :] * y1 + h11[None, :] * step[None, :] * m1
    out = out.astype(np.float32, copy=False)
    return out[0] if squeeze else out


def maybe_resample_curve_matrix(
    curves: np.ndarray,
    freq_axis_ghz: np.ndarray,
    target_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return curves and frequency axis, optionally resampled to target_points."""

    target_points = int(target_points or 0)
    freq_axis_ghz = np.asarray(freq_axis_ghz, dtype=np.float32)
    if target_points <= 0 or curves.shape[1] == target_points:
        return curves.astype(np.float32, copy=False), freq_axis_ghz
    target_axis = uniform_freq_axis(float(freq_axis_ghz[0]), float(freq_axis_ghz[-1]), target_points)
    return pchip_interpolate_rows(freq_axis_ghz, curves, target_axis), target_axis
