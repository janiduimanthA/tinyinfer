"""Plot benchmark results: time per generated token vs context length."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Fixed color + marker per variant, so each keeps its look as new variants are added.
# Colors are a colorblind-safe categorical palette, assigned in a fixed order.
STYLE = {
    "full":      {"color": "#257de0", "marker": "o", "label": "Full logits (baseline)"},
    "last_only": {"color": "#f47230", "marker": "s", "label": "Last-position logits"},
    "kv_cache":  {"color": "#00b894", "marker": "^", "label": "KV cache"},
    "cpp":       {"color": "#f2a600", "marker": "D", "label": "C++"},
}
TEXT = "#333333"
MUTED = "#8a8a8a"


def main():
    data = json.loads(Path("benchmarks/reference_numpy.json").read_text())
    results = data["results"]
    contexts = [r["context"] for r in results]
    variants = [k for k in results[0] if k != "context"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in variants:
        s = STYLE.get(name, {"color": MUTED, "marker": "x", "label": name})
        ys = [r[name] for r in results]
        ax.plot(contexts, ys, color=s["color"], marker=s["marker"],
                markersize=6, linewidth=2, label=s["label"])
        # Direct label at the end of each line, in text color (not the series color)
        ax.annotate(f"{ys[-1]:.0f} ms", (contexts[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=TEXT, fontsize=9)

    ax.set_title("tinyinfer NumPy reference: time per generated token",
                 color=TEXT, loc="left")
    ax.set_xlabel("Context length (tokens)", color=TEXT)
    ax.set_ylabel("Milliseconds per token (lower is better)", color=TEXT)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path("docs/bench_step_time.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()