from __future__ import annotations

from claim_file_splitter.models import make_decision
from claim_file_splitter.pipeline import build_segments


def test_build_segments_forces_confident_type_change_boundary() -> None:
    decisions = [
        make_decision(1, "other", True, confidence=0.2),
        make_decision(2, "repair_invoices", False, confidence=0.8),
    ]

    segments = build_segments(decisions)

    assert len(segments) == 2
    assert segments[0]["document_type"] == "other"
    assert segments[1]["document_type"] == "repair_invoices"
