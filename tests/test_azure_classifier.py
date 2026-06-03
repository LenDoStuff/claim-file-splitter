from __future__ import annotations

import json
from types import SimpleNamespace

from claim_file_splitter.classifiers import azure_classify_pages
from claim_file_splitter.customization import BatchClassificationOutput
from claim_file_splitter.customization import PageDecisionOutput


def test_azure_classifier_uses_responses_parse_with_structured_output() -> None:
    last_request = {}
    parsed_output = BatchClassificationOutput(
        pages=[
            PageDecisionOutput(
                page=1,
                document_type="police_reports",
                starts_new_document=True,
                title="Police Report",
                confidence=0.88,
                reason="Crash report and officer signals.",
            )
        ]
    )
    fake_client = fake_openai_client(parsed_output, last_request)

    decisions = azure_classify_pages(
        fake_client,
        "claims-model",
        [page(1, "POLICE REPORT\nOfficer notes")],
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
    assert last_request["text_format"] is BatchClassificationOutput

    request_text = json.dumps(last_request, default=str)
    assert "POLICE REPORT" not in request_text
    assert "data:image/jpeg;base64," in request_text

    user_content = last_request["input"][1]["content"]
    assert user_content[0]["type"] == "input_text"
    assert user_content[1]["type"] == "input_image"
    assert user_content[1]["detail"] == "high"
    assert fake_client.responses.create_called is False
    assert fake_client.chat.completions.create_called is False


def fake_openai_client(parsed_output: BatchClassificationOutput, last_request: dict):
    def parse(**kwargs):
        last_request.update(kwargs)
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
