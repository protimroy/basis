"""
Utilities for reading TensorBoard event files and plotting training metrics
inline in notebooks, so runs are visible without a TensorBoard server.
"""

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> dict:
    """
    Load all scalar data from a single TensorBoard run directory.

    Returns
    -------
    dict
        ``{tag: {"steps": [...], "values": [...]}}``
    """
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps":  [e.step  for e in events],
            "values": [e.value for e in events],
        }
    return data


def load_latest_run(runs_dir: str) -> tuple:
    """
    Load the most recently created ``run_*`` subdirectory.

    Returns
    -------
    tuple[dict, str]
        ``(data, run_path)``
    """
    runs = sorted(glob.glob(os.path.join(runs_dir, "run_*")))
    if not runs:
        # Also accept the bare directory itself (e.g. nn_training_gpu_cuda)
        runs = sorted(glob.glob(os.path.join(runs_dir, "*")))
    if not runs:
        raise FileNotFoundError(f"No runs found under {runs_dir!r}")
    latest = runs[-1]
    return load_run(latest), latest


def load_all_runs(runs_dir: str) -> dict:
    """
    Load every ``run_*`` subdirectory under *runs_dir*.

    Returns
    -------
    dict
        ``{run_name: data_dict}``
    """
    runs = sorted(glob.glob(os.path.join(runs_dir, "run_*")))
    return {Path(r).name: load_run(r) for r in runs}


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------

def _plot_scalar(data, tag, ax, xlabel="Epoch", ylabel=None, title=None,
                 color=None, label=None):
    """Plot a single scalar tag onto *ax*. Returns True if data existed."""
    if tag not in data:
        return False
    ax.plot(data[tag]["steps"], data[tag]["values"],
            color=color, label=label, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or tag.split("/")[-1])
    ax.set_title(title or tag)
    ax.grid(True, alpha=0.3)
    return True


def plot_loss(data: dict, ax=None, label: str = None,
              title: str = "Training Loss") -> plt.Axes:
    """
    Plot ``Loss/epoch`` from *data*.

    Parameters
    ----------
    data : dict
        Output of :func:`load_run` / :func:`load_latest_run`.
    ax : matplotlib.Axes, optional
        Existing axes to draw on; a new figure is created when omitted.
    label : str, optional
        Legend label (useful when overlaying multiple runs).
    title : str
        Axes title.

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    found = _plot_scalar(data, "Loss/epoch", ax,
                         ylabel="Loss", title=title, label=label)
    if not found:
        ax.set_title(f"{title} (no data)")
    if label:
        ax.legend()
    return ax


def plot_timing(data: dict, axes=None) -> np.ndarray:
    """
    Plot epoch-time and throughput side-by-side.

    Parameters
    ----------
    data : dict
        Output of :func:`load_run`.
    axes : array-like of 2 matplotlib.Axes, optional
        Existing axes pair; a new figure is created when omitted.

    Returns
    -------
    numpy.ndarray of matplotlib.Axes  (shape 2,)
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_scalar(data, "Timing/epoch_time_ms", axes[0],
                 ylabel="Time (ms)", title="Epoch Time", color="tab:blue")
    _plot_scalar(data, "Timing/samples_per_second", axes[1],
                 ylabel="Samples / s", title="Throughput", color="tab:orange")
    return np.asarray(axes)


def plot_pass_timing(data: dict, axes=None) -> np.ndarray:
    """
    Plot per-pass forward/backward timing.

    Parameters
    ----------
    data : dict
        Output of :func:`load_run`.
    axes : array-like of 2 matplotlib.Axes, optional

    Returns
    -------
    numpy.ndarray of matplotlib.Axes  (shape 2,)
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_scalar(data, "Timing/forward_pass_ms", axes[0],
                 xlabel="Step", ylabel="Time (ms)", title="Forward Pass Time",
                 color="steelblue")
    _plot_scalar(data, "Timing/backward_pass_ms", axes[1],
                 xlabel="Step", ylabel="Time (ms)", title="Backward Pass Time",
                 color="tomato")
    return np.asarray(axes)


def plot_gpu_memory(data: dict, ax=None) -> plt.Axes:
    """
    Plot GPU memory metrics (allocated, reserved, free) on a single axes.

    Parameters
    ----------
    data : dict
        Output of :func:`load_run`.
    ax : matplotlib.Axes, optional

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    tags = {
        "GPU/memory_allocated_mb": ("Allocated", "tab:blue"),
        "GPU/memory_reserved_mb":  ("Reserved",  "tab:orange"),
        "GPU/memory_free_mb":      ("Free",      "tab:green"),
    }
    any_found = False
    for tag, (label, color) in tags.items():
        if _plot_scalar(data, tag, ax, ylabel="Memory (MB)",
                        title="GPU Memory", color=color, label=label):
            any_found = True
    if any_found:
        ax.legend()
    else:
        ax.set_title("GPU Memory (no data)")
    return ax


def plot_gpu_utilization(data: dict, axes=None) -> np.ndarray:
    """
    Plot GPU utilization, temperature, and power draw.

    Parameters
    ----------
    data : dict
        Output of :func:`load_run`.
    axes : array-like of 3 matplotlib.Axes, optional

    Returns
    -------
    numpy.ndarray of matplotlib.Axes  (shape 3,)
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_scalar(data, "GPU/utilization_percent", axes[0],
                 ylabel="%", title="GPU Utilization", color="tab:purple")
    _plot_scalar(data, "GPU/temperature_celsius", axes[1],
                 ylabel="°C", title="GPU Temperature", color="tab:red")
    _plot_scalar(data, "GPU/power_draw_watts", axes[2],
                 ylabel="Watts", title="GPU Power Draw", color="tab:brown")
    return np.asarray(axes)


def plot_system(data: dict, ax=None) -> plt.Axes:
    """
    Plot system CPU usage.

    Parameters
    ----------
    data : dict
        Output of :func:`load_run`.
    ax : matplotlib.Axes, optional

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    _plot_scalar(data, "System/cpu_percent", ax,
                 ylabel="%", title="CPU Utilization", color="tab:cyan")
    return ax


# ---------------------------------------------------------------------------
# Full dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(data: dict, run_name: str = "", figsize: tuple = (16, 20)):
    """
    Render a full dashboard of every available metric group from *data*.

    The figure is divided into rows that are shown only when the corresponding
    data tags are present, so the layout adapts to CPU-only runs automatically.

    Parameters
    ----------
    data : dict
        Output of :func:`load_run` or :func:`load_latest_run`.
    run_name : str
        Title prefix (e.g. run directory name).
    figsize : tuple
        ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.Figure
    """
    has_loss   = "Loss/epoch" in data
    has_timing = ("Timing/epoch_time_ms" in data or
                  "Timing/samples_per_second" in data)
    has_pass   = ("Timing/forward_pass_ms" in data or
                  "Timing/backward_pass_ms" in data)
    has_gpumem = any(t in data for t in ("GPU/memory_allocated_mb",
                                          "GPU/memory_reserved_mb",
                                          "GPU/memory_free_mb"))
    has_gpuutil = any(t in data for t in ("GPU/utilization_percent",
                                           "GPU/temperature_celsius",
                                           "GPU/power_draw_watts"))
    has_sys    = "System/cpu_percent" in data

    # Build row layout: each entry is (ncols, label)
    rows = []
    if has_loss:                      rows.append((1,  "loss"))
    if has_timing:                    rows.append((2,  "timing"))
    if has_pass:                      rows.append((2,  "pass"))
    if has_gpumem or has_gpuutil:     rows.append((3,  "gpu"))
    if has_sys:                       rows.append((1,  "sys"))

    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No scalar data found", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    nrows = len(rows)
    max_cols = max(r[0] for r in rows)
    fig = plt.figure(figsize=figsize)
    title = f"Training Dashboard — {run_name}" if run_name else "Training Dashboard"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(nrows, max_cols, figure=fig,
                           hspace=0.5, wspace=0.4)
    row_idx = 0

    for ncols, label in rows:
        if label == "loss":
            ax = fig.add_subplot(gs[row_idx, :])
            plot_loss(data, ax=ax)

        elif label == "timing":
            axes = [fig.add_subplot(gs[row_idx, i]) for i in range(2)]
            plot_timing(data, axes=axes)

        elif label == "pass":
            axes = [fig.add_subplot(gs[row_idx, i]) for i in range(2)]
            plot_pass_timing(data, axes=axes)

        elif label == "gpu":
            # Memory spans 2 cols, util spans remaining col
            ax_mem  = fig.add_subplot(gs[row_idx, :2])
            ax_util = fig.add_subplot(gs[row_idx, 2])
            if has_gpumem:
                plot_gpu_memory(data, ax=ax_mem)
            if has_gpuutil:
                _plot_scalar(data, "GPU/utilization_percent", ax_util,
                             ylabel="%", title="GPU Utilization",
                             color="tab:purple")

        elif label == "sys":
            ax = fig.add_subplot(gs[row_idx, :])
            plot_system(data, ax=ax)

        row_idx += 1

    plt.tight_layout()
    return fig
