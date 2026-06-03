from __future__ import annotations

from claim_file_splitter.cli import _build_parser, load_cli_environment


def test_load_cli_environment_populates_parser_defaults(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "AZURE_AI_PROJECT_ENDPOINT=https://example.services.ai.azure.com/api/projects/demo",
                "AZURE_OPENAI_DEPLOYMENT=claims-vision-model",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    load_cli_environment(["--env-file", str(env_file)])
    args = _build_parser().parse_args(["claim.pdf"])

    assert (
        args.project_endpoint
        == "https://example.services.ai.azure.com/api/projects/demo"
    )
    assert args.deployment == "claims-vision-model"
