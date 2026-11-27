"""Plot batching results: throughput and latency vs batch size (two panels, one axis each)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

BLUE = "#257de0"
TEXT = "#333333"
MUTED = "#8a8a8a"
GRID = "#e6e6e6"


def style(ax, title, ylabel, xticks):
    ax.set_title(title, color=TEXT, loc="left", fontsize=11)
    ax.set_xlabel("Batch size (sequences decoded together)", color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_xticks(xticks)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main():
    data = json.loads(Path("benchmarks/batch.json").read_text())
    rows = data["decode"]
    batch = [r["batch"] for r in rows]
    tput = [r["throughput_tok_s"] for r in rows]
    step = [r["step_ms"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: throughput, with the ideal "perfect batching" line for reference
    ax1.plot(batch, [tput[0] * b for b in batch], color=MUTED, linestyle="--",
             linewidth=1.5, label="Ideal (perfect scaling)")
    ax1.plot(batch, tput, color=BLUE, marker="o", markersize=6, linewidth=2, label="Measured")
    ax1.annotate(f"{tput[-1]:.0f} tok/s", (batch[-1], tput[-1]), xytext=(6, 0),
                 textcoords="offset points", va="center", color=TEXT, fontsize=9)
    style(ax1, "Throughput: total tokens per second", "Tokens / second (higher is better)", batch)
    ax1.legend(frameon=False)

    # Right: latency of one decode step (what each user waits per token)
    ax2.plot(batch, step, color=BLUE, marker="o", markersize=6, linewidth=2)
    ax2.annotate(f"{step[-1]:.0f} ms", (batch[-1], step[-1]), xytext=(6, 0),
                 textcoords="offset points", va="center", color=TEXT, fontsize=9)
    style(ax2, "Latency: time per decode step", "Milliseconds per step (lower is better)", batch)

    fig.suptitle(f"tinyinfer NumPy reference: batched decoding at context {data['context']}",
                 color=TEXT, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()

    out = Path("docs/bench_batch.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()