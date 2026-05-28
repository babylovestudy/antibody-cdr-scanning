"""Tests for CDR mutation scanning engine."""

import numpy as np
import pytest

from cdr_scan.antibodies import ANTIBODIES
from cdr_scan.engine import AMINO_ACIDS, CDRScanner, MutationResult


class TestMutationResult:
    def test_to_dict(self):
        r = MutationResult(1, "W", "A", -1.5, -8.0, -6.5, 0.3, 0.8)
        d = r.to_dict()
        assert d["position"] == 1
        assert d["wildtype"] == "W"
        assert d["mutant"] == "A"
        assert d["disruption_score"] == 0.8

    def test_same_aa_raises_or_skips(self):
        """Wildtype == mutant should not appear in scan results."""
        r1 = MutationResult(1, "W", "A", -1.5, -8.0, -6.5, 0.3, 0.8)
        assert r1.wildtype != r1.mutant


class TestCDRScanner:
    @pytest.fixture(scope="class")
    def scanner(self):
        return CDRScanner(
            model_name="facebook/esm2_t6_8M_UR50D", device="cpu"
        )

    def test_init_loads_model(self, scanner):
        assert scanner.model is not None
        assert scanner.tokenizer is not None

    def test_mask_token_exists(self, scanner):
        assert scanner._mask_token_id is not None

    def test_aa_token_ids(self, scanner):
        for aa in AMINO_ACIDS:
            assert aa in scanner._aa_to_token

    def test_scan_cdr_returns_all_mutations(self, scanner):
        """A 5-aa CDR should produce 5 × 19 = 95 mutations."""
        seq = "AAAAA"
        results = scanner.scan_cdr(seq, "test")
        assert len(results) == 5 * 19

    def test_scan_cdr_sorted_by_disruption(self, scanner):
        seq = "ACDEF"
        results = scanner.scan_cdr(seq, "test")
        scores = [r.disruption_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_embed_sequence_shape(self, scanner):
        emb = scanner.embed_sequence("ACDEFGHIK")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (320,)  # ESM-2 8M dim

    def test_embedding_distance_same_seq(self, scanner):
        emb = scanner.embed_sequence("ACDEFGHIK")
        dist = scanner.embedding_distance(emb, "ACDEFGHIK")
        assert dist < 0.001  # Identical should be ~0

    def test_embedding_distance_different_seq(self, scanner):
        emb = scanner.embed_sequence("ACDEFGHIK")
        dist = scanner.embedding_distance(emb, "XXXXXXXXXX")
        assert dist > 0.0  # Different should be > 0

    def test_mutation_result_has_all_fields(self, scanner):
        seq = "WGGDG"
        results = scanner.scan_cdr(seq, "test")
        r = results[0]
        assert r.position > 0
        assert r.wildtype in AMINO_ACIDS
        assert r.mutant in AMINO_ACIDS
        assert isinstance(r.delta_logprob, float)
        assert isinstance(r.embedding_distance, float)
        assert 0.0 <= r.disruption_score <= 1.0


class TestAntibodyData:
    def test_herceptin_has_all_cdrs(self):
        ab = ANTIBODIES["herceptin"]
        assert len(ab.h1) > 0
        assert len(ab.h2) > 0
        assert len(ab.h3) > 0
        assert len(ab.hotspot_h3) > 0

    def test_known_hotspots_are_1_indexed(self):
        ab = ANTIBODIES["herceptin"]
        for hp in ab.hotspot_h3:
            assert 1 <= hp <= len(ab.h3)

    def test_all_antibodies_have_names(self):
        for key, ab in ANTIBODIES.items():
            assert ab.name
            assert ab.target
            assert len(ab.h3) > 0  # H3 is the most important CDR
