from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from claim_file_splitter import split_claim_file_azure
from claim_file_splitter.models import ClaimSplitResult


def test_default_config_preserves_current_folder_behavior(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_sample_claim_pdf(source_pdf)

    output_dir = tmp_path / "output"
    result = split_claim_file_azure(
        source_pdf,
        output_dir=output_dir,
        deployment="claims-model",
        batch_size=2,
        client=fake_openai_client(
            {
                1: decision("repair_invoices", True, title="Page 1"),
                2: decision("repair_invoices", False, title="Page 2"),
                3: decision("appraisals", True, title="Page 3"),
                4: decision("communications", True, title="Page 4"),
                5: decision("payments", True, title="Page 5"),
                6: decision("legal_correspondence", True, title="Page 6"),
            }
        ),
    )

    assert isinstance(result, ClaimSplitResult)
    assert [document.document_type for document in result.documents] == [
        "repair_invoices",
        "appraisals",
        "communications",
        "payments",
        "legal_correspondence",
    ]
    assert [
        (document.start_page, document.end_page)
        for document in result.documents
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
    assert set(manifest) == {
        "source_pdf",
        "output_dir",
        "document_count",
        "documents",
    }
    assert manifest["document_count"] == 5
    assert manifest["documents"][0] == {
        "document_id": 1,
        "name": "Page 1",
        "summary": "Page 1 Test classifier decision.",
        "path": str(repair_pdf),
        "document_type": "repair_invoices",
        "start_page": 1,
        "end_page": 2,
        "page_count": 2,
        "confidence": 0.9,
    }


def test_direct_categories_replace_defaults_and_filename_prefixes(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_sample_claim_pdf(source_pdf)
    result = split_claim_file_azure(
        source_pdf,
        output_dir=tmp_path / "output",
        deployment="claims-model",
        categories=[
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
        default_document_type="misc",
        batch_size=10,
        client=fake_openai_client(
            {
                1: decision("shop_bills", True),
                2: decision("shop_bills", False),
                3: decision("misc", True),
                4: decision("misc", False),
                5: decision("misc", False),
                6: decision("misc", False),
            }
        ),
    )

    assert {document.document_type for document in result.documents} == {
        "shop_bills",
        "misc",
    }
    assert (tmp_path / "output" / "shop_bills" / "shop_bill_001.pdf").exists()
    assert (tmp_path / "output" / "misc" / "misc_doc_001.pdf").exists()
    assert not (tmp_path / "output" / "repair_invoices").exists()


def test_config_batch_size_changes_batch_grouping(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_numbered_claim_pdf(source_pdf, page_count=5)
    batches = []

    split_claim_file_azure(
        source_pdf,
        output_dir=tmp_path / "output",
        deployment="claims-model",
        batch_size=2,
        client=fake_openai_client(
            {
                1: decision("other", True),
                2: decision("other", False),
                3: decision("other", False),
                4: decision("other", False),
                5: decision("other", False),
            },
            batches=batches,
        ),
    )

    assert batches == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_multi_page_invoice_is_written_as_one_original_pdf(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim_file.pdf"
    _write_numbered_claim_pdf(source_pdf, page_count=4)
    result = split_claim_file_azure(
        source_pdf,
        output_dir=tmp_path / "output",
        deployment="claims-model",
        categories=[
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
        default_document_type="other",
        batch_size=5,
        client=fake_openai_client(
            {
                1: decision("repair_invoices", True, title="Repair Invoice"),
                2: decision("repair_invoices", False, title="Repair Invoice"),
                3: decision("repair_invoices", False, title="Repair Invoice"),
                4: decision("payments", True, title="Payment"),
            }
        ),
    )

    invoice = result.documents[0]
    payment = result.documents[1]
    assert invoice.document_type == "repair_invoices"
    assert invoice.page_count == 3
    assert invoice.summary == "Repair Invoice Test classifier decision."
    assert payment.document_type == "payments"
    assert payment.page_count == 1
    assert payment.summary == "Payment Test classifier decision."

    invoice_reader = PdfReader(str(invoice.path))
    payment_reader = PdfReader(str(payment.path))
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


def decision(
    document_type: str,
    starts_new_document: bool,
    *,
    title: str = "",
    confidence: float = 0.9,
    reason: str = "Test classifier decision.",
) -> dict:
    return {
        "document_type": document_type,
        "starts_new_document": starts_new_document,
        "title": title,
        "confidence": confidence,
        "reason": reason,
    }


def fake_openai_client(
    decisions_by_page: dict[int, dict],
    *,
    batches: list[list[int]] | None = None,
):
    def parse(**kwargs):
        prompt = json.loads(kwargs["input"][1]["content"][0]["text"])
        if batches is not None:
            batches.append([page["page"] for page in prompt["target_pages"]])
        parsed_pages = [
            {
                "page": page["page"],
                **decisions_by_page[page["page"]],
            }
            for page in prompt["target_pages"]
        ]
        parsed_output = kwargs["text_format"].model_validate({"pages": parsed_pages})
        return SimpleNamespace(output_parsed=parsed_output)

    return SimpleNamespace(responses=SimpleNamespace(parse=parse))
