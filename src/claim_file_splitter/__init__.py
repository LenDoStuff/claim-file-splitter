"""Azure claim-file PDF splitting module."""

from .customization import ClaimSplitterConfig
from .models import ClaimSplitResult, DocumentSegment, PageDecision, WrittenDocument
from .pipeline import split_claim_file_azure

__all__ = [
    "ClaimSplitResult",
    "ClaimSplitterConfig",
    "DocumentSegment",
    "PageDecision",
    "WrittenDocument",
    "split_claim_file_azure",
]
