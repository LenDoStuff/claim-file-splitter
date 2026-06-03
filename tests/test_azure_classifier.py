from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from claim_file_splitter.classifiers import AzureProjectPageClassifier
from claim_file_splitter.models import PageFeatures, PageImage


def test_azure_classifier_parses_openai_responses_output_text() -> None:
    fake_client = _FakeOpenAIClient(
        {
            "pages": [
                {
                    "page": 1,
                    "document_type": "police_report",
                    "starts_new_document": True,
                    "title": "Police Report",
                    "confidence": 0.88,
                    "reason": "Crash report and officer signals.",
                }
            ]
        }
    )
    classifier = AzureProjectPageClassifier(
        project_endpoint="https://example.services.ai.azure.com/api/projects/demo",
        deployment="claims-model",
        client=fake_client,
    )

    decisions = classifier.classify_pages([_page(1, "POLICE REPORT\nOfficer notes")])

    assert len(decisions) == 1
    assert decisions[0].document_type == "police_reports"
    assert decisions[0].starts_new_document is True
    assert decisions[0].confidence == 0.88
    assert fake_client.responses.last_request["model"] == "claims-model"
    request_text = json.dumps(fake_client.responses.last_request)
    assert "POLICE REPORT" not in request_text
    assert "input_image" in request_text
    assert "data:image/jpeg;base64," in request_text


class _FakeOpenAIClient:
    def __init__(self, payload: dict) -> None:
        self.responses = _FakeResponses(payload)


class _FakeResponses:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_request: dict | None = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(output_text=json.dumps(self.payload))


def _page(page_number: int, text: str) -> PageFeatures:
    return PageFeatures(
        source_path=Path("claim.pdf"),
        page_number=page_number,
        text=text,
        word_count=len(text.split()),
        char_count=len(text),
        image_count=0,
        is_image_only=False,
        may_require_ocr=False,
        image=PageImage(
            page_number=page_number,
            mime_type="image/jpeg",
            width_px=100,
            height_px=200,
            byte_size=12,
            data_uri="data:image/jpeg;base64,AAAA",
        ),
    )
