from __future__ import annotations

from claim_file_splitter.customization import resolve_config


def test_resolve_config_uses_defaults_and_direct_overrides(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://env.example")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")

    config = resolve_config(
        project_endpoint="https://arg.example",
        deployment="arg-deployment",
        temperature=0.2,
        batch_size=2,
        recent_page_decision_limit=3,
        completed_document_limit=1,
        high_confidence_batch_boundary=0.8,
        other_type_boundary_confidence=0.7,
        type_change_boundary_confidence=0.6,
    )

    assert config.azure.project_endpoint == "https://arg.example"
    assert config.azure.deployment == "arg-deployment"
    assert config.azure.temperature == 0.2
    assert config.splitter.batch_size == 2
    assert config.splitter.recent_page_decision_limit == 3
    assert config.splitter.completed_document_limit == 1
    assert config.splitter.high_confidence_batch_boundary == 0.8
    assert config.splitter.other_type_boundary_confidence == 0.7
    assert config.splitter.type_change_boundary_confidence == 0.6
    assert config.rendering.image_detail == "high"


def test_resolve_config_ignores_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://env.example")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")

    config = resolve_config()

    assert config.azure.project_endpoint is None
    assert config.azure.deployment is None
