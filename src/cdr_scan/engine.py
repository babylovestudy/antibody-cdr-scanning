"""Mutation scanning engine using ESM-2 for antibody CDR analysis.

Two complementary scoring methods:

1. Pseudo-perplexity (masked prediction):
   Mask each position, compare ESM-2's predicted probability for
   the wildtype AA vs. each mutant AA. A high wildtype probability
   and low mutant probability → mutation likely disruptive.

2. Embedding distance (cosine similarity):
   Embed wildtype and mutant sequences, measure cosine distance.
   Larger distance → mutation changes the protein's representation
   more → potentially more disruptive.

Reference: Meier et al. (2021) "Language models enable zero-shot prediction
of the effects of mutations on protein function" (NeurIPS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class MutationResult:
    position: int          # 1-indexed position in CDR
    wildtype: str          # Original amino acid
    mutant: str            # Mutant amino acid
    wt_logprob: float      # ESM-2 log probability of wildtype AA at this position
    mutant_logprob: float  # ESM-2 log probability of mutant AA at this position
    delta_logprob: float   # mutant_logprob - wt_logprob (positive = mutant favored)
    embedding_distance: float  # Cosine distance between wt and mutant embeddings
    disruption_score: float    # Combined score (higher = more disruptive)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "wildtype": self.wildtype,
            "mutant": self.mutant,
            "wt_logprob": round(self.wt_logprob, 4),
            "mutant_logprob": round(self.mutant_logprob, 4),
            "delta_logprob": round(self.delta_logprob, 4),
            "embedding_distance": round(self.embedding_distance, 4),
            "disruption_score": round(self.disruption_score, 4),
        }


class CDRScanner:
    """Scan all single-point mutations of a CDR sequence using ESM-2.

    Parameters
    ----------
    model_name : str
        ESM-2 model (e.g., facebook/esm2_t6_8M_UR50D).
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "cpu",
    ):
        from transformers import AutoTokenizer, EsmForMaskedLM

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()

        # Amino acid token IDs for ESM-2
        self._aa_to_token: dict[str, int] = {}
        for aa in AMINO_ACIDS:
            token_id = self.tokenizer.convert_tokens_to_ids(aa)
            self._aa_to_token[aa] = token_id

        self._mask_token_id = self.tokenizer.mask_token_id

    # ------------------------------------------------------------------
    # Method 1: Pseudo-perplexity (masked prediction)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_by_masking(
        self, sequence: str, context: str = ""
    ) -> dict[int, dict[str, float]]:
        """Score all 19 mutations at each position by masked prediction.

        For each position i:
          1. Create a masked version: seq[:i] + <mask> + seq[i+1:]
          2. Run ESM-2, get logits at the masked position
          3. Compute log-probability for each of the 20 AAs

        Returns:
            dict: position (1-indexed) → {AA: log_probability}
        """
        seq = sequence.upper()
        full_seq = context + seq if context else seq
        seq_start = len(context) if context else 0
        seq_len = len(seq)

        # Tokenize
        encoded = self.tokenizer(
            full_seq, return_tensors="pt", padding=False, truncation=True,
            max_length=1024,
        )
        input_ids = encoded["input_ids"].to(self.device)  # (1, L_full)

        # Map sequence positions to token positions
        # ESM-2 uses BOS/EOS tokens; find the token positions for our CDR
        tokens = input_ids[0]
        token_strs = self.tokenizer.convert_ids_to_tokens(tokens)

        # Find the token range corresponding to the CDR sequence
        # Simple approach: try matching the first few residues
        cdr_token_start = seq_start + 1  # +1 for BOS

        results: dict[int, dict[str, float]] = {}

        for i in range(seq_len):
            pos_1idx = i + 1
            mask_pos = cdr_token_start + i

            # Create masked input
            masked_ids = input_ids.clone()
            masked_ids[0, mask_pos] = self._mask_token_id

            # Forward pass with EsmForMaskedLM — returns logits directly
            outputs = self.model(masked_ids)
            logits = outputs.logits[0, mask_pos, :]  # (vocab_size,)

            # Log probabilities for all AAs
            log_probs = F.log_softmax(logits, dim=-1)
            aa_scores: dict[str, float] = {}
            for aa in AMINO_ACIDS:
                token_id = self._aa_to_token.get(aa)
                if token_id is not None:
                    aa_scores[aa] = float(log_probs[token_id].cpu())

            results[pos_1idx] = aa_scores

        return results

    # ------------------------------------------------------------------
    # Method 2: Embedding distance
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Get mean-pooled ESM-2 embedding for a sequence."""
        encoded = self.tokenizer(
            sequence, return_tensors="pt", padding=False, truncation=True,
            max_length=1024,
        ).to(self.device)
        # EsmForMaskedLM wraps EsmModel; use base model for embeddings
        outputs = self.model.esm(**encoded)
        hidden = outputs.last_hidden_state  # (1, L, D)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled[0].cpu().numpy()

    def embedding_distance(
        self, wt_emb: np.ndarray, mutant_seq: str
    ) -> float:
        """Cosine distance between wildtype and mutant embeddings."""
        mut_emb = self.embed_sequence(mutant_seq)
        cos_sim = float(
            np.dot(wt_emb, mut_emb)
            / (np.linalg.norm(wt_emb) * np.linalg.norm(mut_emb))
        )
        return 1.0 - cos_sim  # 0 = identical, 2 = maximally different

    # ------------------------------------------------------------------
    # Full scan
    # ------------------------------------------------------------------

    def scan_cdr(
        self,
        cdr_sequence: str,
        cdr_name: str = "CDR",
        context_sequence: str = "",
    ) -> list[MutationResult]:
        """Scan all single-point mutations in a CDR sequence.

        Args:
            cdr_sequence: amino acid sequence of the CDR loop
            cdr_name: label (e.g., "H3")
            context_sequence: surrounding sequence for context window
                              (improves masked prediction accuracy)

        Returns:
            List of MutationResult for all 19 × len(cdr) mutations,
            sorted by disruption_score descending.
        """
        seq = cdr_sequence.upper()
        seq_len = len(seq)

        # Get masked prediction scores
        context = context_sequence.upper() if context_sequence else ""
        mask_scores = self.score_by_masking(seq, context)

        # Get wildtype embedding (with context)
        full_wt = context + seq
        wt_emb = self.embed_sequence(full_wt)

        # Scan all mutations
        results: list[MutationResult] = []

        for i in range(seq_len):
            pos_1idx = i + 1
            wt_aa = seq[i]

            if wt_aa not in AMINO_ACIDS:
                continue

            wt_logprob = mask_scores.get(pos_1idx, {}).get(wt_aa, 0.0)

            for mut_aa in AMINO_ACIDS:
                if mut_aa == wt_aa:
                    continue

                mutant_seq = seq[:i] + mut_aa + seq[i + 1:]
                mut_logprob = mask_scores.get(pos_1idx, {}).get(mut_aa, -10.0)

                # Disruption: positive = mutant worse than wildtype
                delta = mut_logprob - wt_logprob  # typically negative for bad mutations

                # Embedding distance
                full_mut = context + mutant_seq
                emb_dist = self.embedding_distance(wt_emb, full_mut)

                # Combined disruption score
                # Normalize delta_logprob to [0, 1] scale
                # Typical delta range: -8 to 2
                prob_score = max(0.0, min(1.0, -delta / 5.0))
                emb_score = max(0.0, min(1.0, emb_dist * 5.0))
                disruption = 0.5 * prob_score + 0.5 * emb_score

                results.append(
                    MutationResult(
                        position=pos_1idx,
                        wildtype=wt_aa,
                        mutant=mut_aa,
                        wt_logprob=wt_logprob,
                        mutant_logprob=mut_logprob,
                        delta_logprob=delta,
                        embedding_distance=emb_dist,
                        disruption_score=disruption,
                    )
                )

        results.sort(key=lambda r: r.disruption_score, reverse=True)
        return results

    def scan_antibody_cdrs(
        self,
        antibody_cdr,
        context_sequence: str = "",
    ) -> dict[str, list[MutationResult]]:
        """Scan all CDR loops of an antibody.

        Returns:
            dict: {cdr_name: [MutationResult, ...]}
        """
        results: dict[str, list[MutationResult]] = {}
        for cdr_name in ["h1", "h2", "h3", "l1", "l2", "l3"]:
            seq = getattr(antibody_cdr, cdr_name, "")
            if seq:
                print(f"  Scanning {cdr_name.upper()} ({len(seq)} aa)...")
                results[cdr_name] = self.scan_cdr(seq, cdr_name, context_sequence)
        return results

    def summary_table(
        self,
        results: list[MutationResult],
        top_k: int = 20,
    ) -> str:
        """Format a human-readable summary table."""
        lines = [
            f"{'Pos':>4s} {'WT':>3s} {'Mut':>3s} "
            f"{'ΔLogP':>8s} {'EmbDist':>8s} {'Disruption':>10s}"
        ]
        lines.append("-" * 44)
        for r in results[:top_k]:
            lines.append(
                f"{r.position:4d} {r.wildtype:>3s} {r.mutant:>3s} "
                f"{r.delta_logprob:8.4f} {r.embedding_distance:8.4f} "
                f"{r.disruption_score:10.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hotspot validation
# ---------------------------------------------------------------------------

def validate_against_hotspots(
    results: list[MutationResult],
    known_hotspots: list[int],
    cdr_name: str = "",
) -> dict[str, float]:
    """Compare predicted top-disruption mutations against known hotspots.

    A good prediction: known hotspot positions should have high disruption scores.

    Returns:
        dict with:
          - hotspot_avg_rank: mean rank of hotspot positions among all positions
          - hotspot_avg_score: mean disruption score at hotspot vs non-hotspot
          - enrichment: fraction of top-K mutations at hotspot positions
    """
    if not known_hotspots:
        return {}

    # Per-position average disruption
    pos_scores: dict[int, float] = {}
    for r in results:
        pos_scores[r.position] = max(
            pos_scores.get(r.position, 0.0), r.disruption_score
        )

    positions = sorted(pos_scores.keys())
    sorted_positions = sorted(positions, key=lambda p: pos_scores[p], reverse=True)

    hotspot_ranks = []
    for hp in known_hotspots:
        if hp in pos_scores:
            rank = sorted_positions.index(hp) + 1
            hotspot_ranks.append(rank)

    hotspot_mean = float(
        np.mean([pos_scores.get(hp, 0.0) for hp in known_hotspots])
    )
    non_hotspot_mean = float(
        np.mean(
            [pos_scores[p] for p in positions if p not in known_hotspots]
        )
    )

    # Enrichment: what fraction of top-N mutations are at hotspots?
    n_hotspots = len([hp for hp in known_hotspots if hp in pos_scores])
    if n_hotspots > 0:
        top_n_positions = set(sorted_positions[:n_hotspots])
        hotspot_hits = sum(1 for hp in known_hotspots if hp in top_n_positions)
        enrichment = hotspot_hits / n_hotspots
    else:
        enrichment = 0.0

    return {
        "hotspot_avg_rank": float(np.mean(hotspot_ranks)) if hotspot_ranks else 0.0,
        "hotspot_avg_score": hotspot_mean,
        "non_hotspot_avg_score": non_hotspot_mean,
        "enrichment": enrichment,
        "cdr_name": cdr_name,
    }
