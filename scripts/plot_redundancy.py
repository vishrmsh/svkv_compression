"""Plot paired contrasts from aggregate-only closure results; no clinical inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="results/redundancy")
    args = parser.parse_args()
    root = Path(args.directory)
    labels = [("keydiff", "KeyDiff"), ("linear_leverage", "Exact linear ridge leverage"),
              ("kernel_leverage", "Exact RBF kernel ridge leverage")]
    specifications = [("synthetic", "recall_rare", "Rare-group recall", "contexts"),
                      ("clinical", "retain", "ICU event-hour retention", "stays")]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.4), constrained_layout=True)
    input_hashes = {}
    for ax, (name, metric, title, unit) in zip(axes, specifications):
        source = root / f"{name}_summary.json"
        report = json.loads(source.read_text())
        input_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        for y, (method, _) in enumerate(labels):
            delta = report["svdd_minus_comparator"][method][metric]
            mean = 100 * delta["mean"]
            low, high = [100 * v for v in delta["bootstrap_95_ci"]]
            ax.plot([low, high], [y, y], color="#315b9d", linewidth=2.2)
            ax.plot(mean, y, "o", color="#315b9d", markersize=7)
            ax.annotate(f"{mean:+.1f} [{low:+.1f}, {high:+.1f}]", (mean, y),
                        xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10)
        ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(labels)), [label for _, label in labels])
        ax.set_ylim(2.5, -.7)
        ax.margins(x=.22)
        ax.set_title(f"{title} · {report['complete']} {unit} · {100 * report['mean_budget']:.1f}% mean retention",
                     loc="left", fontsize=12, pad=12)
        ax.set_xlabel("SVDD − comparator (percentage points)")
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Original selection tasks: fixed protocols, reused data\nPositive favors SVDD; negative favors the comparator",
                 fontsize=14)
    fig.supxlabel("Paired 95% percentile bootstrap intervals; no multiplicity adjustment.\nICU SpO₂ is withheld from selectors. These are selection probes, not KV-cache benchmarks.",
                  fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"paired_contrasts.{suffix}", dpi=180)
    plt.close(fig)
    (root / "plot_manifest.json").write_text(json.dumps({
        "input_sha256": input_hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib": matplotlib.__version__,
        "artifacts": ["paired_contrasts.png", "paired_contrasts.pdf"],
        "data_scope": "aggregate summaries only; no patient-level inputs",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
