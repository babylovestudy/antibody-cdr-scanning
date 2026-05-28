#!/usr/bin/env python
"""Run CDR mutation scanning on therapeutic antibodies.

Usage:
    python -m cdr_scan.run --antibody herceptin --cdr h3
    python -m cdr_scan.run --antibody all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .antibodies import ANTIBODIES
from .engine import CDRScanner, validate_against_hotspots
from .visualize import plot_hotspot_validation, plot_mutation_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan antibody CDR mutations using ESM-2"
    )
    parser.add_argument(
        "--antibody", default="herceptin",
        choices=list(ANTIBODIES.keys()) + ["all"],
        help="Which antibody to scan",
    )
    parser.add_argument(
        "--cdr", default="h3",
        choices=["h1", "h2", "h3", "l1", "l2", "l3"],
        help="Which CDR loop to scan (ignored if --antibody all)",
    )
    parser.add_argument(
        "--model", default="facebook/esm2_t6_8M_UR50D",
        help="ESM-2 model name",
    )
    parser.add_argument(
        "--device", default="cpu",
    )
    parser.add_argument(
        "--out-dir", default="outputs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scanner = CDRScanner(model_name=args.model, device=args.device)

    antibodies_to_scan = (
        list(ANTIBODIES.values()) if args.antibody == "all"
        else [ANTIBODIES[args.antibody]]
    )

    all_results: dict[str, dict] = {}

    for ab in antibodies_to_scan:
        print(f"\n{'='*60}")
        print(f"Scanning: {ab.name} ({ab.target}, PDB: {ab.pdb})")
        print(f"{'='*60}")

        cdrs_to_scan = (
            ["h1", "h2", "h3", "l1", "l2", "l3"]
            if args.antibody == "all"
            else [args.cdr]
        )

        ab_results: dict = {"name": ab.name, "target": ab.target, "cdrs": {}}

        for cdr_name in cdrs_to_scan:
            seq = getattr(ab, cdr_name, "")
            if not seq:
                continue

            print(f"\n  CDR-{cdr_name.upper()}: {seq}")
            results = scanner.scan_cdr(seq, cdr_name)

            # Print top 15 most disruptive mutations
            print(scanner.summary_table(results, top_k=15))

            # Save all results
            cdr_out = out_dir / f"{args.antibody}_{cdr_name}_scan.csv"
            pd.DataFrame([r.to_dict() for r in results]).to_csv(
                cdr_out, index=False
            )
            print(f"\n  Full results saved to {cdr_out}")

            # Plot
            hm_path = str(out_dir / f"{args.antibody}_{cdr_name}_heatmap.png")
            plot_mutation_heatmap(results, f"{ab.name} CDR-{cdr_name.upper()}", hm_path)
            print(f"  Heatmap saved to {hm_path}")

            # Validate against known hotspots
            hotspots = getattr(ab, f"hotspot_{cdr_name}", [])
            if hotspots:
                validation = validate_against_hotspots(results, hotspots, cdr_name)
                print(
                    f"  Hotspot validation: "
                    f"enrichment={validation['enrichment']:.2f}, "
                    f"hotspot_avg_rank={validation['hotspot_avg_rank']:.1f}, "
                    f"hotspot_score={validation['hotspot_avg_score']:.3f} "
                    f"vs non_hotspot={validation['non_hotspot_avg_score']:.3f}"
                )

                bar_path = str(out_dir / f"{args.antibody}_{cdr_name}_hotspots.png")
                plot_hotspot_validation(results, hotspots, f"{ab.name} CDR-{cdr_name.upper()}", bar_path)
                print(f"  Hotspot plot saved to {bar_path}")

                ab_results["cdrs"][cdr_name] = {
                    "sequence": seq,
                    "n_mutations": len(results),
                    "validation": validation,
                }
            else:
                ab_results["cdrs"][cdr_name] = {
                    "sequence": seq,
                    "n_mutations": len(results),
                }

        all_results[ab.name] = ab_results

    # Save summary JSON
    with open(out_dir / "scan_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSummary saved to {out_dir}/scan_summary.json")


if __name__ == "__main__":
    main()
