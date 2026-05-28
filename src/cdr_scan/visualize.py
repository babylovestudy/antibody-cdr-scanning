"""Visualization for CDR mutation scanning results."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine import MutationResult

AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")


def plot_mutation_heatmap(
    results: list[MutationResult],
    cdr_name: str = "CDR",
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> None:
    """Plot a heatmap: positions × amino acids, colored by disruption score.

    Wildtype residues are marked with a black border.
    """
    # Build matrix: (n_positions × 20 amino acids)
    positions = sorted(set(r.position for r in results))
    n_pos = len(positions)
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    matrix = np.full((n_pos, 20), np.nan)
    wt_labels = [""] * n_pos

    for r in results:
        i = pos_to_idx[r.position]
        j = aa_to_idx[r.mutant]
        matrix[i, j] = r.disruption_score
        if not wt_labels[i]:
            wt_labels[i] = r.wildtype

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.T, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)

    # Mark wildtype positions
    for i in range(n_pos):
        if wt_labels[i] in AMINO_ACIDS:
            j = aa_to_idx[wt_labels[i]]
            ax.add_patch(
                plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False,
                              edgecolor="black", linewidth=2)
            )

    ax.set_xticks(range(n_pos))
    ax.set_xticklabels(
        [f"{p}\n{wt_labels[i]}" for i, p in enumerate(positions)],
        fontsize=8,
    )
    ax.set_yticks(range(20))
    ax.set_yticklabels(AMINO_ACIDS)
    ax.set_xlabel("Position (wildtype)")
    ax.set_ylabel("Mutant Amino Acid")
    ax.set_title(f"Mutation Disruption Heatmap: {cdr_name}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Disruption Score (higher = more disruptive)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_hotspot_validation(
    results: list[MutationResult],
    known_hotspots: list[int],
    cdr_name: str = "CDR",
    save_path: str | None = None,
) -> None:
    """Bar chart: per-position average disruption, highlighting known hotspots."""
    positions = sorted(set(r.position for r in results))

    # Per-position mean disruption
    pos_scores: dict[int, float] = {}
    for r in results:
        pos_scores.setdefault(r.position, []).append(r.disruption_score)

    means = [float(np.mean(pos_scores.get(p, [0]))) for p in positions]
    colors = [
        "coral" if p in known_hotspots else "steelblue"
        for p in positions
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(positions, means, color=colors)

    ax.set_xlabel("Position")
    ax.set_ylabel("Mean Disruption Score")
    ax.set_title(f"Per-Position Disruption: {cdr_name} (red = known hotspot)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="coral", label="Known hotspot (literature)"),
        Patch(facecolor="steelblue", label="Other position"),
    ]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
