from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from claim_file_splitter.classifiers import RuleBasedPageClassifier
from claim_file_splitter.models import PageDecision
from claim_file_splitter.pipeline import ClaimFileSplitter, SplitterConfig


def test_rule_based_pipeline_splits_and_organizes_claim_pdf(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_sample_claim_pdf(source_pdf)

    output_dir = tmp_path / "output"
    splitter = ClaimFileSplitter(
        classifier=RuleBasedPageClassifier(),
        config=SplitterConfig(
            output_dir=output_dir,
            batch_size=2,
            use_pdfplumber_fallback=False,
        ),
    )

    result = splitter.run(source_pdf)

    assert [segment.document_type for segment in result.segments] == [
        "repair_invoices",
        "appraisals",
        "communications",
        "payments",
        "legal_correspondence",
    ]
    assert [(segment.start_page, segment.end_page) for segment in result.segments] == [
        (1, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
    ]

    repair_pdf = output_dir / "repair_invoices" / "repair_invoice_001.pdf"
    appraisal_pdf = output_dir / "appraisals" / "appraisal_001.pdf"
    communication_pdf = output_dir / "communications" / "communication_001.pdf"
    payment_pdf = output_dir / "payments" / "payment_document_001.pdf"
    legal_pdf = output_dir / "legal_correspondence" / "legal_correspondence_001.pdf"

    for output_pdf in (
        repair_pdf,
        appraisal_pdf,
        communication_pdf,
        payment_pdf,
        legal_pdf,
    ):
        assert output_pdf.exists()

    assert len(PdfReader(str(repair_pdf)).pages) == 2
    assert len(PdfReader(str(appraisal_pdf)).pages) == 1
    assert len(PdfReader(str(communication_pdf)).pages) == 1
    assert len(PdfReader(str(payment_pdf)).pages) == 1
    assert len(PdfReader(str(legal_pdf)).pages) == 1

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["page_count"] == 6
    assert manifest["document_count"] == 5
    assert manifest["documents"][0]["output_path"].endswith("repair_invoice_001.pdf")


def test_image_pipeline_uses_five_page_batches_and_rolling_context(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_numbered_claim_pdf(source_pdf, page_count=11)

    classifier = _ImageBatchClassifier()
    output_dir = tmp_path / "output"
    splitter = ClaimFileSplitter(
        classifier=classifier,
        config=SplitterConfig(
            output_dir=output_dir,
            batch_size=5,
            render_dpi=100,
            keep_page_images=True,
            use_pdfplumber_fallback=False,
        ),
    )

    result = splitter.run(source_pdf)

    assert classifier.calls == [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11],
    ]
    assert classifier.contexts[0]["open_document"] is None
    assert classifier.contexts[1]["open_document"]["document_type"] == "repair_invoices"
    assert classifier.contexts[1]["open_document"]["start_page"] == 4
    assert classifier.contexts[1]["open_document"]["end_page"] == 5
    assert classifier.contexts[2]["open_document"]["document_type"] == "communications"
    assert classifier.contexts[2]["open_document"]["start_page"] == 8
    assert classifier.contexts[2]["open_document"]["end_page"] == 10

    assert [(segment.document_type, segment.start_page, segment.end_page) for segment in result.segments] == [
        ("appraisals", 1, 3),
        ("repair_invoices", 4, 7),
        ("communications", 8, 10),
        ("payments", 11, 11),
    ]
    assert (output_dir / "repair_invoices" / "repair_invoice_001.pdf").exists()
    assert len(PdfReader(str(output_dir / "repair_invoices" / "repair_invoice_001.pdf")).pages) == 4

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["classification_batches"][1]["page_numbers"] == [6, 7, 8, 9, 10]
    assert "inherited from page 5" in manifest["page_decisions"][5]["reason"]
    assert manifest["pages"][0]["rendered_image"]["path"].endswith("page_000001.jpg")


def _write_sample_claim_pdf(path: Path) -> None:
    pages = [
        [
            "AUTO REPAIR INVOICE",
            "Invoice Number: RI-1001",
            "Parts, labor, body shop amount due for vehicle repair.",
        ],
        [
            "Repair continuation",
            "Parts and labor detail continued for the same invoice.",
        ],
        [
            "APPRAISAL REPORT",
            "Vehicle value and estimate of damages after collision.",
        ],
        [
            "EMAIL THREAD",
            "From: adjuster@example.com",
            "To: claimant@example.com",
            "Subject: Claim status update",
        ],
        [
            "PAYMENT NOTICE",
            "Settlement payment issued. Check Number: 987654.",
        ],
        [
            "LAW OFFICE OF EXAMPLE AND SMITH",
            "Demand letter from attorney and legal counsel.",
        ],
    ]

    pdf = canvas.Canvas(str(path), pagesize=letter)
    for page_lines in pages:
        y = 740
        for line in page_lines:
            pdf.drawString(72, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()


def _write_numbered_claim_pdf(path: Path, page_count: int) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    for page_number in range(1, page_count + 1):
        pdf.drawString(72, 740, f"Claim file page {page_number}")
        pdf.showPage()
    pdf.save()


class _ImageBatchClassifier:
    requires_page_images = True

    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.contexts: list[dict] = []

    def classify_pages(
        self,
        pages,
        *,
        previous_page=None,
        rolling_context=None,
    ) -> list[PageDecision]:
        self.calls.append([page.page_number for page in pages])
        self.contexts.append(rolling_context or {})
        assert all(page.image is not None for page in pages)

        decisions: list[PageDecision] = []
        for page in pages:
            page_number = page.page_number
            if page_number <= 3:
                decisions.append(
                    PageDecision(
                        page_number,
                        "appraisals",
                        page_number == 1,
                        confidence=0.92,
                        reason="appraisal",
                    )
                )
            elif page_number in {4, 5, 7}:
                decisions.append(
                    PageDecision(
                        page_number,
                        "repair_invoices",
                        page_number == 4,
                        confidence=0.9,
                        reason="repair",
                    )
                )
            elif page_number == 6:
                decisions.append(
                    PageDecision(
                        page_number,
                        "other",
                        False,
                        confidence=0.35,
                        reason="continued page with weak local signal",
                    )
                )
            elif page_number <= 10:
                decisions.append(
                    PageDecision(
                        page_number,
                        "communications",
                        page_number == 8,
                        confidence=0.89,
                        reason="communication",
                    )
                )
            else:
                decisions.append(
                    PageDecision(
                        page_number,
                        "payments",
                        True,
                        confidence=0.93,
                        reason="payment",
                    )
                )
        return decisions
