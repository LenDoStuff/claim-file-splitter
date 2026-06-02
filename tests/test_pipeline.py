from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from claim_file_splitter.classifiers import RuleBasedPageClassifier
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
