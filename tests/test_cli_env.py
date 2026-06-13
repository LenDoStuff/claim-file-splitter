from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import claim_file_splitter.cli as cli
from claim_file_splitter.cli import build_parser, load_cli_environment, main
from claim_file_splitter.customization import resolve_config


def test_load_cli_environment_populates_resolved_config(tmp_path, monkeypatch) -> None:
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
    config = resolve_config()

    assert (
        config.azure.project_endpoint
        == "https://example.services.ai.azure.com/api/projects/demo"
    )
    assert config.azure.deployment == "claims-vision-model"


def test_direct_args_override_config_file_and_env(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "splitter.json"
    config_path.write_text(
        json.dumps(
            {
                "azure": {
                    "project_endpoint": "https://file.example",
                    "deployment": "file-deployment",
                },
                "splitter": {"batch_size": 4},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", "https://env.example")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")

    config = resolve_config(
        config_path=config_path,
        project_endpoint="https://arg.example",
        deployment="arg-deployment",
        batch_size=2,
    )

    assert config.azure.project_endpoint == "https://arg.example"
    assert config.azure.deployment == "arg-deployment"
    assert config.splitter.batch_size == 2


def test_cli_config_loads_json_and_flags_override_values(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    config_path = tmp_path / "splitter.json"
    config_path.write_text(
        json.dumps({"splitter": {"batch_size": 3}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    captured = {}

    def fake_split_claim_file_azure(input_pdf, **kwargs):
        captured["input_pdf"] = input_pdf
        captured.update(kwargs)
        return SimpleNamespace(
            source_pdf=Path(input_pdf),
            output_dir=Path(kwargs["output_dir"]),
            manifest_path=Path(kwargs["output_dir"]) / "manifest.json",
            page_count=0,
            document_count=1,
            documents=[
                SimpleNamespace(
                    segment=SimpleNamespace(
                        document_type="repair_invoices",
                        start_page=1,
                        end_page=2,
                        summary="Repair invoice summary.",
                    ),
                    output_path=Path(kwargs["output_dir"])
                    / "repair_invoices"
                    / "repair_invoice_001.pdf",
                )
            ],
        )

    monkeypatch.setattr(cli, "split_claim_file_azure", fake_split_claim_file_azure)

    exit_code = main(
        [
            "claim.pdf",
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--batch-size",
            "1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["documents"][0]["summary"] == "Repair invoice summary."
    assert captured["config_path"] == config_path
    assert captured["batch_size"] == 1
    assert captured["output_dir"] == str(output_dir)


def test_build_parser_accepts_config_option() -> None:
    args = build_parser().parse_args(["claim.pdf", "--config", "splitter.json"])

    assert args.config == Path("splitter.json")
