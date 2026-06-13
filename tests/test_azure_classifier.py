from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from claim_file_splitter import ClaimSplitResult, split_claim_file_azure
from claim_file_splitter.classifiers import azure_classify_pages
from claim_file_splitter.customization import ClaimSplitterConfig
from claim_file_splitter.customization import make_batch_classification_output_model


def test_azure_classifier_uses_responses_parse_with_structured_output() -> None:
    config = ClaimSplitterConfig.model_validate(
        {"rendering": {"image_detail": "high"}}
    )
    last_request = {}
    fake_client = fake_openai_client(last_request)

    decisions = azure_classify_pages(
        fake_client,
        "claims-model",
        [page(1, "POLICE REPORT\nOfficer notes")],
        config=config,
    )

    assert decisions == [
        {
            "page_number": 1,
            "document_type": "police_reports",
            "starts_new_document": True,
            "title": "Police Report",
            "confidence": 0.88,
            "reason": "Crash report and officer signals.",
        }
    ]
    assert last_request["model"] == "claims-model"
    assert last_request["text_format"].__name__ == "BatchClassificationOutput"

    request_text = json.dumps(last_request, default=str)
    assert "POLICE REPORT" not in request_text
    assert "data:image/jpeg;base64," in request_text

    user_content = last_request["input"][1]["content"]
    assert user_content[0]["type"] == "input_text"
    assert user_content[1]["type"] == "input_image"
    assert user_content[1]["detail"] == "high"
    assert fake_client.responses.create_called is False
    assert fake_client.chat.completions.create_called is False


def test_dynamic_structured_output_accepts_only_configured_categories() -> None:
    config = ClaimSplitterConfig.model_validate(
        {
            "categories": [
                {"name": "bills", "filename_prefix": "bill"},
                {"name": "misc", "filename_prefix": "misc"},
            ],
            "default_document_type": "misc",
        }
    )
    text_format = make_batch_classification_output_model(config)

    parsed = text_format.model_validate(
        {
            "pages": [
                {
                    "page": 1,
                    "document_type": "bills",
                    "starts_new_document": True,
                    "title": "Bill",
                    "confidence": 0.9,
                    "reason": "Configured type.",
                }
            ]
        }
    )
    assert parsed.pages[0].document_type.value == "bills"

    with pytest.raises(ValidationError):
        text_format.model_validate(
            {
                "pages": [
                    {
                        "page": 1,
                        "document_type": "repair_invoices",
                        "starts_new_document": True,
                        "title": "Invoice",
                        "confidence": 0.9,
                        "reason": "Not configured.",
                    }
                ]
            }
        )


def test_public_azure_api_returns_typed_result_with_injected_client(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "claim.pdf"
    _write_pdf(source_pdf)
    last_request = {}
    fake_client = fake_openai_client(last_request)

    result = split_claim_file_azure(
        source_pdf,
        output_dir=tmp_path / "output",
        deployment="claims-model",
        client=fake_client,
        use_pdfplumber_fallback=False,
    )

    assert isinstance(result, ClaimSplitResult)
    assert result.document_count == 1
    assert result.documents[0].segment.document_type == "police_reports"
    assert result.documents[0].output_path.exists()


def fake_openai_client(last_request: dict):
    def parse(**kwargs):
        last_request.update(kwargs)
        parsed_output = kwargs["text_format"].model_validate(
            {
                "pages": [
                    {
                        "page": 1,
                        "document_type": "police_reports",
                        "starts_new_document": True,
                        "title": "Police Report",
                        "confidence": 0.88,
                        "reason": "Crash report and officer signals.",
                    }
                ]
            }
        )
        return SimpleNamespace(output_parsed=parsed_output)

    responses = SimpleNamespace(parse=parse, create_called=False)
    chat_completions = SimpleNamespace(create_called=False)
    chat = SimpleNamespace(completions=chat_completions)
    return SimpleNamespace(responses=responses, chat=chat)


def page(page_number: int, text: str) -> dict:
    return {
        "source_path": "claim.pdf",
        "page_number": page_number,
        "text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
        "image_count": 0,
        "is_image_only": False,
        "may_require_ocr": False,
        "image": {
            "page_number": page_number,
            "mime_type": "image/jpeg",
            "width_px": 100,
            "height_px": 200,
            "byte_size": 12,
            "data_uri": "data:image/jpeg;base64,AAAA",
            "path": None,
        },
    }


def _write_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(72, 740, "POLICE REPORT")
    pdf.showPage()
    pdf.save()
