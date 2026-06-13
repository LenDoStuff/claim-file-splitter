from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from claim_file_splitter.customization import ClaimSplitterConfig
from claim_file_splitter.models import ClaimSplitResult
from claim_file_splitter.models import make_decision
from claim_file_splitter.customization import resolve_config
from claim_file_splitter.pipeline import run_split_pipeline


def test_default_config_preserves_current_folder_behavior(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_sample_claim_pdf(source_pdf)

    output_dir = tmp_path / "output"
    config = resolve_config(
        batch_size=2,
    )
    result = run_split_pipeline(
        source_pdf,
        output_dir=output_dir,
        config=config,
        classify_pages=sample_claim_classifier(config),
        requires_page_images=False,
    )

    assert isinstance(result, ClaimSplitResult)
    assert [segment.document_type for segment in result.segments] == [
        "repair_invoices",
        "appraisals",
        "communications",
        "payments",
        "legal_correspondence",
    ]
    assert [
        (segment.start_page, segment.end_page)
        for segment in result.segments
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

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["page_count"] == 6
    assert manifest["document_count"] == 5
    assert manifest["documents"][0]["summary"] == (
        "Page 1 Test classifier decision."
    )
    assert manifest["documents"][0]["output_path"].endswith("repair_invoice_001.pdf")


def test_json_config_replaces_categories_and_filename_prefixes(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_sample_claim_pdf(source_pdf)
    config_path = tmp_path / "splitter.json"
    config_path.write_text(
        json.dumps(
            {
                "categories": [
                    {
                        "name": "shop_bills",
                        "filename_prefix": "shop_bill",
                        "description": "Configured shop bills.",
                    },
                    {
                        "name": "misc",
                        "filename_prefix": "misc_doc",
                        "description": "Configured fallback.",
                    },
                ],
                "default_document_type": "misc",
                "splitter": {"batch_size": 10},
            }
        ),
        encoding="utf-8",
    )

    config = resolve_config(
        config_path=config_path,
    )
    result = run_split_pipeline(
        source_pdf,
        output_dir=tmp_path / "output",
        config=config,
        classify_pages=configured_category_classifier(config),
        requires_page_images=False,
    )

    assert {segment.document_type for segment in result.segments} == {
        "shop_bills",
        "misc",
    }
    assert (tmp_path / "output" / "shop_bills" / "shop_bill_001.pdf").exists()
    assert (tmp_path / "output" / "misc" / "misc_doc_001.pdf").exists()
    assert not (tmp_path / "output" / "repair_invoices").exists()


def test_config_batch_size_changes_batch_grouping(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_numbered_claim_pdf(source_pdf, page_count=5)
    config = ClaimSplitterConfig.model_validate(
        {"splitter": {"batch_size": 2}}
    )

    result = run_split_pipeline(
        source_pdf,
        output_dir=tmp_path / "output",
        config=config,
        classify_pages=single_document_classifier(config),
        requires_page_images=False,
    )

    assert [batch.page_numbers for batch in result.classification_batches] == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_multi_page_invoice_is_written_as_one_original_pdf(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_numbered_claim_pdf(source_pdf, page_count=4)
    config_path = tmp_path / "splitter.json"
    config_path.write_text(
        json.dumps(
            {
                "categories": [
                    {
                        "name": "repair_invoices",
                        "filename_prefix": "repair_invoice",
                    },
                    {
                        "name": "payments",
                        "filename_prefix": "payment_document",
                    },
                    {
                        "name": "other",
                        "filename_prefix": "document",
                    },
                ],
                "default_document_type": "other",
            }
        ),
        encoding="utf-8",
    )

    config = resolve_config(
        config_path=config_path,
        batch_size=5,
    )
    result = run_split_pipeline(
        source_pdf,
        output_dir=tmp_path / "output",
        config=config,
        classify_pages=invoice_payment_classifier(config),
        requires_page_images=False,
    )

    invoice = result.documents[0]
    payment = result.documents[1]
    assert invoice.segment.document_type == "repair_invoices"
    assert invoice.segment.page_count == 3
    assert invoice.segment.summary == "Repair Invoice Test classifier decision."
    assert payment.segment.document_type == "payments"
    assert payment.segment.page_count == 1
    assert payment.segment.summary == "Payment Test classifier decision."

    invoice_reader = PdfReader(str(invoice.output_path))
    payment_reader = PdfReader(str(payment.output_path))
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
    _write_text_pdf(path, pages)


def _write_numbered_claim_pdf(path: Path, page_count: int) -> None:
    _write_text_pdf(
        path,
        [[f"Claim file page {page_number}"] for page_number in range(1, page_count + 1)],
    )


def _write_text_pdf(path: Path, pages: list[list[str]]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    for page_lines in pages:
        y = 740
        for line in page_lines:
            pdf.drawString(72, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()


def sample_claim_classifier(config: ClaimSplitterConfig):
    page_types = {
        1: ("repair_invoices", True),
        2: ("repair_invoices", False),
        3: ("appraisals", True),
        4: ("communications", True),
        5: ("payments", True),
        6: ("legal_correspondence", True),
    }

    def classify_pages(batch, *, previous_page=None, rolling_context=None):
        return [
            make_decision(
                page["page_number"],
                page_types[page["page_number"]][0],
                page_types[page["page_number"]][1],
                config=config,
                title=f"Page {page['page_number']}",
                confidence=0.9,
                reason="Test classifier decision.",
            )
            for page in batch
        ]

    return classify_pages


def configured_category_classifier(config: ClaimSplitterConfig):
    def classify_pages(batch, *, previous_page=None, rolling_context=None):
        return [
            make_decision(
                page["page_number"],
                "shop_bills" if page["page_number"] <= 2 else "misc",
                page["page_number"] in {1, 3},
                config=config,
                confidence=0.9,
                reason="Test classifier decision.",
            )
            for page in batch
        ]

    return classify_pages


def single_document_classifier(config: ClaimSplitterConfig):
    def classify_pages(batch, *, previous_page=None, rolling_context=None):
        return [
            make_decision(
                page["page_number"],
                "other",
                page["page_number"] == 1,
                config=config,
                confidence=0.8,
                reason="Test classifier decision.",
            )
            for page in batch
        ]

    return classify_pages


def invoice_payment_classifier(config: ClaimSplitterConfig):
    def classify_pages(batch, *, previous_page=None, rolling_context=None):
        return [
            make_decision(
                page["page_number"],
                "repair_invoices" if page["page_number"] <= 3 else "payments",
                page["page_number"] in {1, 4},
                config=config,
                title="Repair Invoice" if page["page_number"] <= 3 else "Payment",
                confidence=0.95,
                reason="Test classifier decision.",
            )
            for page in batch
        ]

    return classify_pages
