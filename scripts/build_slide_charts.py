"""Rebuild the Reveal.js results charts from the project report and frozen scores."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/slides/charts"
INK, MUTED, GREEN, GRID = "#14221f", "#506257", "#18634e", "#c4cbbf"
GOLD, LIGHT = "#ffd376", "#f7f3e4"
FIGURE_SIZE = (10, 3.9)
PLOT_MARGINS = {"left": .23, "right": .95, "bottom": .15, "top": .85}


def report_rows(report, header):
    """Read a results table by its columns, independent of section numbering."""
    _, found, body = report.partition(header + "\n")
    if not found:
        raise ValueError(f"Missing report table: {header}")
    table = body.split("\n\n", 1)[0]
    return [line.strip("| ").split(" | ") for line in table.splitlines()[1:]]


def axes_style(ax, dark=False):
    ink, grid = (LIGHT, "#50645b") if dark else (INK, GRID)
    ax.tick_params(axis="both", colors=ink, length=0, pad=8)
    ax.spines[:].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=grid, linewidth=0.8)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, transparent=True)
    plt.close(fig)


def model_changes(history):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(**PLOT_MARGINS)
    steps = {(s["frm"], s["to"]): s for s in history["steps"]}
    changes = [(("overlap-512", "overlap-256"), "256 px training tiles"),
               (("R1 paired", "Run 21"), "Covariance pooling"),
               (("Run 21", "Run 22 (shipped)"), "Trained cell head")]
    axes_style(ax)
    ax.axvline(0, color=INK, lw=1.3)
    for y, (model, _) in enumerate(changes):
        diff = steps[model]["standard_pe"]["difference"]
        p, lo, hi = (100 * diff[k] for k in ("point", "low", "high"))
        ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="o", color=GREEN,
                    markersize=9, capsize=5, lw=2)
        ax.annotate(f"{p:+.2f}", (hi, y), xytext=(9, 0), textcoords="offset points",
                    va="center", color=INK, weight="bold",
                    bbox={"facecolor": "#f5f2e8", "edgecolor": "none", "pad": 1})
    ax.set(xlim=(-2.7, 1.05), ylim=(2.5, -.5), yticks=range(3),
           yticklabels=[label for _, label in changes], xticks=[-2, -1, 0, 1])
    save(fig, "model-changes.png")


def compare_models(history):
    models = {m["label"]: m for m in history["models"]}
    hard = {}
    for step in history["steps"]:
        for key, side in [("frm", "a"), ("to", "b")]:
            hard[step[key]] = step["hard_k1_low_auc"]["models"][side]["point"]
    labels = ["V3c baseline", "Run 21", "Run 22 (shipped)"]
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, sharey=True)
    fig.subplots_adjust(**PLOT_MARGINS, wspace=.45)
    panels = [
        ("Standard tile\nerror", [100 * models[k]["standard_pe"] for k in labels], (0, 5.4), [0, 2, 4]),
        ("ALASKA unseen\nerror", [100 * models[k]["alaska-unseen_pe"] for k in labels], (0, 36), [0, 15, 30]),
        ("Low-payload\nAUC", [hard[k] for k in labels], (.5, .93), [.5, .65, .8]),
    ]
    for idx, (ax, (title, values, limits, ticks)) in enumerate(zip(axes, panels)):
        axes_style(ax)
        ax.set_title(title, fontsize=13.5, loc="left", color=INK, weight="bold", pad=5)
        ax.set(xlim=limits, ylim=(2.5, -.5), xticks=ticks)
        if idx < 2:
            ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        for y, val in enumerate(values):
            ax.plot(val, y, "o", markersize=9, color=GREEN if y == 2 else MUTED)
            label = f"{val:.2f}%" if idx < 2 else f"{val:.3f}"
            ax.annotate(label, (val, y), xytext=(0, 7), textcoords="offset points",
                        color=INK, fontsize=13, ha="center", va="bottom")
    axes[0].set_yticks(range(3), ["V3c baseline", "Run 21", "Run 22 · shipped"])
    save(fig, "model-comparison.png")


def noise_effect(report):
    rows = report_rows(report, "| Panel | k=1, all bpp | k=1, low bpp | k=1, low bpp + noisy |")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(**PLOT_MARGINS)
    axes_style(ax, dark=True)
    ax.axvline(.5, color=GOLD, lw=1.5, linestyle="--")
    for y, (_, _, low_cell, noisy_cell) in enumerate(rows):
        low, noisy = (float(cell.split("/")[0]) for cell in (low_cell, noisy_cell))
        ax.plot([noisy, low], [y, y], color="#94aa9e", lw=2)
        ax.plot(low, y, "o", color=LIGHT, markersize=9)
        ax.plot(noisy, y, "D", color=GOLD, markersize=9)
        for val, offset, align, color in [(noisy, -9, "right", GOLD), (low, 9, "left", LIGHT)]:
            ax.annotate(f"{val:.3f}", (val, y), xytext=(offset, 7),
                        textcoords="offset points", ha=align, color=color, fontsize=14,
                        bbox={"facecolor": INK, "edgecolor": "none", "pad": 1})
    ax.set(xlim=(.45, 1.02), ylim=(len(rows)-.5, -.7), yticks=range(len(rows)),
           yticklabels=[r[0] for r in rows], xticks=[.5, .6, .7, .8, .9, 1])
    save(fig, "noise-effect.png")


def photo_rules(report):
    rows = report_rows(report, "| Rule | False alarms | Misses | P_E |")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(**PLOT_MARGINS)
    axes_style(ax)
    for y, row in enumerate(rows):
        rates = [float(cell.strip("*%")) for cell in row[1:4]]
        for val, offset, color in zip(rates, [-.23, 0, .23], ["#58736c", "#ac5829", GREEN]):
            ax.barh(y + offset, val, height=.17, color=color)
            ax.annotate(f"{val:.1f}%", (val, y + offset), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=13)
    ax.set(xlim=(0, 72), ylim=(2.55, -.55), yticks=range(3),
           yticklabels=["Tiles only", "Cells only", "Tiles + cells"], xticks=[0, 20, 40, 60])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    save(fig, "photo-rules.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans"})
    history = json.loads((ROOT / "docs/results/model-history-2026-11-25.json").read_text())
    report = (ROOT / "README.md").read_text(encoding="utf-8")
    model_changes(history)
    compare_models(history)
    noise_effect(report)
    photo_rules(report)
