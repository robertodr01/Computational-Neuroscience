"""Tools for simulating and visualizing Izhikevich neuron dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class IzhikevichConfig:
    """Container for Izhikevich model coefficients."""

    a: float
    b: float
    c: float
    d: float
    v0: float
    u0: float | None = None
    k_v: float = 5.0
    k_0: float = 140.0


@dataclass
class Trace:
    """Simulation output."""

    time: np.ndarray
    v: np.ndarray
    u: np.ndarray
    current: np.ndarray


def simulate(
    cfg: IzhikevichConfig,
    current_fn: Callable[[float], float],
    t_stop: float = 200.0,
    dt: float = 0.25,
    spike_threshold: float = 30.0,
    clip_spikes: bool = True,
    recovery_mode: str = "standard",
) -> Trace:
    """Integrate the Izhikevich equations with Euler's method."""

    if dt <= 0:
        raise ValueError("dt must be > 0")

    t = np.arange(0.0, t_stop + dt, dt)
    v = np.empty_like(t)
    u = np.empty_like(t)
    i = np.empty_like(t)

    v_now = cfg.v0
    u_now = cfg.b * cfg.v0 if cfg.u0 is None else cfg.u0

    for idx, time in enumerate(t):
        i_now = float(current_fn(float(time)))
        i[idx] = i_now

        v_now = v_now + dt * (0.04 * v_now * v_now + cfg.k_v * v_now + cfg.k_0 - u_now + i_now)

        if recovery_mode == "accommodation":
            target = cfg.b * (v_now + 65.0)
        else:
            target = cfg.b * v_now
        u_now = u_now + dt * cfg.a * (target - u_now)

        if v_now >= spike_threshold:
            v[idx] = spike_threshold if clip_spikes else v_now
            v_now = cfg.c
            u_now = u_now + cfg.d
        else:
            v[idx] = v_now

        u[idx] = u_now

    return Trace(time=t, v=v, u=u, current=i)


def rectangular_pulse(start: float, stop: float, amplitude: float, baseline: float = 0.0) -> Callable[[float], float]:
    """Single rectangular pulse."""

    def fn(time: float) -> float:
        return amplitude if start <= time <= stop else baseline

    return fn


def pulse_train(intervals: list[tuple[float, float]], amplitude: float, baseline: float = 0.0) -> Callable[[float], float]:
    """Piecewise constant pulse train."""

    def fn(time: float) -> float:
        for start, stop in intervals:
            if start <= time <= stop:
                return amplitude
        return baseline

    return fn


def save_trace_plots(trace: Trace, title: str, output_dir: str | Path) -> None:
    """Save membrane potential and phase portrait."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = (
        title.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    axes[0].plot(trace.time, trace.v, lw=0.9, color="#1f77b4")
    axes[0].set_title(f"{title} - Membrane potential")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("v")
    axes[0].grid(alpha=0.3)

    axes[1].plot(trace.v, trace.u, lw=0.9, color="#d62728")
    axes[1].set_title(f"{title} - Phase portrait")
    axes[1].set_xlabel("v")
    axes[1].set_ylabel("u")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name}.png", dpi=140, bbox_inches="tight")
    plt.show()


def linear_ramp(after: float, slope: float, intercept_before: float = 0.0) -> Callable[[float], float]:
    """Current that grows linearly after a time threshold."""

    def fn(time: float) -> float:
        if time <= after:
            return intercept_before
        return intercept_before + slope * (time - after)

    return fn
