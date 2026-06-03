from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from claim_file_splitter.classifiers import rule_based_classify_pages
from claim_file_splitter.models import make_decision
from claim_file_splitter.pipeline import split_claim_file


def test_rule_based_pipeline_splits_and_organizes_claim_pdf(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_sample_claim_pdf(source_pdf)

    output_dir = tmp_path / "output"
    result = split_claim_file(
        source_pdf,
        output_dir=output_dir,
        classify_pages=rule_based_classify_pages,
        batch_size=2,
        use_pdfplumber_fallback=False,
    )

    assert [segment["document_type"] for segment in result["segments"]] == [
        "repair_invoices",
        "appraisals",
        "communications",
        "payments",
        "legal_correspondence",
    ]
    assert [
        (segment["start_page"], segment["end_page"])
        for segment in result["segments"]
    ] == [
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

    calls: list[list[int]] = []
    contexts: list[dict] = []

    def image_batch_classifier(batch, *, previous_page=None, rolling_context=None):
        calls.append([page["page_number"] for page in batch])
        contexts.append(rolling_context or {})
        assert all(page["image"] is not None for page in batch)
        return [_decision_for_page(page["page_number"]) for page in batch]

    output_dir = tmp_path / "output"
    result = split_claim_file(
        source_pdf,
        output_dir=output_dir,
        classify_pages=image_batch_classifier,
        requires_page_images=True,
        batch_size=5,
        render_dpi=100,
        keep_page_images=True,
        use_pdfplumber_fallback=False,
    )

    assert calls == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11]]
    assert contexts[0]["open_document"] is None
    assert contexts[1]["open_document"]["document_type"] == "repair_invoices"
    assert contexts[1]["open_document"]["start_page"] == 4
    assert contexts[1]["open_document"]["end_page"] == 5
    assert contexts[2]["open_document"]["document_type"] == "communications"
    assert contexts[2]["open_document"]["start_page"] == 8
    assert contexts[2]["open_document"]["end_page"] == 10

    assert [
        (segment["document_type"], segment["start_page"], segment["end_page"])
        for segment in result["segments"]
    ] == [
        ("appraisals", 1, 3),
        ("repair_invoices", 4, 7),
        ("communications", 8, 10),
        ("payments", 11, 11),
    ]

    repair_pdf = output_dir / "repair_invoices" / "repair_invoice_001.pdf"
    assert repair_pdf.exists()
    assert len(PdfReader(str(repair_pdf)).pages) == 4

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["classification_batches"][1]["page_numbers"] == [6, 7, 8, 9, 10]
    assert "inherited from page 5" in manifest["page_decisions"][5]["reason"]
    assert manifest["pages"][0]["rendered_image"]["path"].endswith("page_000001.jpg")


def test_multi_page_invoice_is_written_as_one_pdf(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_numbered_claim_pdf(source_pdf, page_count=4)

    def classifier(batch, *, previous_page=None, rolling_context=None):
        decisions = []
        for page in batch:
            page_number = page["page_number"]
            if page_number <= 3:
                decisions.append(
                    make_decision(
                        page_number,
                        "repair_invoices",
                        page_number == 1,
                        title="Repair Invoice",
                        confidence=0.94,
                        reason="Invoice page.",
                    )
                )
            else:
                decisions.append(
                    make_decision(
                        page_number,
                        "payments",
                        True,
                        title="Payment Notice",
                        confidence=0.91,
                        reason="Payment page.",
                    )
                )
        return decisions

    output_dir = tmp_path / "output"
    split_claim_file(
        source_pdf,
        output_dir=output_dir,
        classify_pages=classifier,
        batch_size=5,
        use_pdfplumber_fallback=False,
    )

    invoice_pdf = output_dir / "repair_invoices" / "repair_invoice_001.pdf"
    payment_pdf = output_dir / "payments" / "payment_document_001.pdf"
    assert invoice_pdf.exists()
    assert payment_pdf.exists()
    invoice_reader = PdfReader(str(invoice_pdf))
    payment_reader = PdfReader(str(payment_pdf))
    assert len(invoice_reader.pages) == 3
    assert len(payment_reader.pages) == 1
    assert [
        (page.extract_text() or "").strip()
        for page in invoice_reader.pages
    ] == [
        "Claim file page 1",
        "Claim file page 2",
        "Claim file page 3",
    ]
    assert (payment_reader.pages[0].extract_text() or "").strip() == (
        "Claim file page 4"
    )


def _decision_for_page(page_number: int) -> dict:
    if page_number <= 3:
        return make_decision(
            page_number,
            "appraisals",
            page_number == 1,
            confidence=0.92,
            reason="appraisal",
        )
    if page_number in {4, 5, 7}:
        return make_decision(
            page_number,
            "repair_invoices",
            page_number == 4,
            confidence=0.9,
            reason="repair",
        )
    if page_number == 6:
        return make_decision(
            page_number,
            "other",
            False,
            confidence=0.35,
            reason="continued page with weak local signal",
        )
    if page_number <= 10:
        return make_decision(
            page_number,
            "communications",
            page_number == 8,
            confidence=0.89,
            reason="communication",
        )
    return make_decision(
        page_number,
        "payments",
        True,
        confidence=0.93,
        reason="payment",
    )


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
