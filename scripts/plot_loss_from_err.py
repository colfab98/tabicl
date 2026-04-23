#!/usr/bin/env python3
"""Plot training cross-entropy curves from TabICL .err logs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


STEP_RE = re.compile(r"Step:\s+\d+%\|.*?\|\s*(\d+)/(?:\d+)\s*\[")
CE_RE = re.compile(r"ce=([A-Za-z0-9.+-]+)")


def parse_err_file(path: Path) -> tuple[list[int], list[float]]:
    text = path.read_text(errors="replace")
    steps: list[int] = []
    losses: list[float] = []

    chunks = re.split(r"[\r\n]+", text)
    for chunk in chunks:
        if "Step:" not in chunk or "ce=" not in chunk:
            continue

        step_match = STEP_RE.search(chunk)
        ce_match = CE_RE.search(chunk)
        if not step_match or not ce_match:
            continue

        step = int(step_match.group(1))
        ce_raw = ce_match.group(1)
        try:
            loss = float(ce_raw)
        except ValueError:
            continue

        if math.isnan(loss) or math.isinf(loss):
            continue

        # Keep the latest value for a repeated progress step.
        if step in steps:
            idx = steps.index(step)
            losses[idx] = loss
        else:
            steps.append(step)
            losses.append(loss)

    return steps, losses


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, min(window, len(values)))
    out: list[float] = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot ce loss curves from TabICL .err files.")
    parser.add_argument("err_files", nargs="+", type=Path, help="One or more .err files to parse")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. If omitted and one .err file is given, use <err_stem>_loss.svg next to the log.",
    )
    parser.add_argument("--title", default="Training Cross-Entropy", help="Plot title")
    parser.add_argument(
        "--trend-window",
        type=int,
        default=25,
        help="Moving-average window for the red trend line (default: 25)",
    )
    return parser


def save_svg(
    series: list[tuple[str, list[int], list[float]]],
    output: Path,
    title: str,
    trend_window: int,
) -> None:
    width = 1000
    height = 500
    left = 70
    right = 20
    top = 40
    bottom = 50
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_steps = [x for _, steps, _ in series for x in steps]
    all_losses = [y for _, _, losses in series for y in losses]
    min_x, max_x = min(all_steps), max(all_steps)
    min_y, max_y = min(all_losses), max(all_losses)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1.0

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="20" font-family="sans-serif">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="{width/2}" y="{height - 10}" text-anchor="middle" font-size="14" font-family="sans-serif">Step</text>',
        f'<text x="18" y="{height/2}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 18 {height/2})">Cross-Entropy (ce)</text>',
    ]

    for i in range(5):
        frac = i / 4
        x = left + frac * plot_w
        y = top + plot_h - frac * plot_h
        x_val = min_x + frac * (max_x - min_x)
        y_val = min_y + frac * (max_y - min_y)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#dddddd"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="12" font-family="sans-serif">{x_val:.0f}</text>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif">{y_val:.3g}</text>'
        )

    for idx, (label, steps, losses) in enumerate(series):
        color = colors[idx % len(colors)]
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(steps, losses))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        trend = moving_average(losses, trend_window)
        trend_points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(steps, trend))
        parts.append(f'<polyline fill="none" stroke="#d62728" stroke-width="2.5" points="{trend_points}"/>')
        legend_y = top + 18 * idx
        parts.append(f'<line x1="{width - 220}" y1="{legend_y}" x2="{width - 200}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        parts.append(
            f'<text x="{width - 195}" y="{legend_y + 4}" font-size="12" font-family="sans-serif">{label}</text>'
        )
        if idx == 0:
            trend_legend_y = top + 18 * (len(series) + 1)
            parts.append(
                f'<line x1="{width - 220}" y1="{trend_legend_y}" x2="{width - 200}" y2="{trend_legend_y}" stroke="#d62728" stroke-width="2.5"/>'
            )
            parts.append(
                f'<text x="{width - 195}" y="{trend_legend_y + 4}" font-size="12" font-family="sans-serif">trend</text>'
            )

    parts.append("</svg>")
    output.write_text("\n".join(parts))


def main() -> None:
    args = build_parser().parse_args()

    series: list[tuple[str, list[int], list[float]]] = []

    plotted = 0
    for err_file in args.err_files:
        steps, losses = parse_err_file(err_file)
        if not steps:
            print(f"No parsable ce values found in {err_file}")
            continue

        print(
            f"Parsed {len(steps)} points from {err_file} "
            f"(step {steps[0]}->{steps[-1]}, ce {losses[0]:.4g}->{losses[-1]:.4g})"
        )
        series.append((err_file.stem, steps, losses))
        plotted += 1

    if plotted == 0:
        raise SystemExit("No valid loss curves found in the provided files.")

    output = args.output
    if output is None:
        if len(args.err_files) == 1:
            err_file = args.err_files[0]
            output = err_file.with_name(f"{err_file.stem}_loss.svg")
        else:
            output = Path("loss_curve.svg")

    if plt is not None:
        plt.figure(figsize=(10, 5))
        for label, steps, losses in series:
            plt.plot(steps, losses, label=label, linewidth=1.8)
            plt.plot(steps, moving_average(losses, args.trend_window), color="red", linewidth=2.2, alpha=0.9)
        plt.xlabel("Step")
        plt.ylabel("Cross-Entropy (ce)")
        plt.title(args.title)
        plt.grid(True, alpha=0.3)
        if plotted > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(output, dpi=160)
    else:
        if output.suffix.lower() not in {".svg"}:
            output = output.with_suffix(".svg")
        save_svg(series, output, args.title, args.trend_window)

    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
