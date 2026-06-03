from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .classifiers import AzureProjectPageClassifier, RuleBasedPageClassifier
from .pipeline import ClaimFileSplitter, SplitterConfig


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    load_cli_environment(argv)
    parser = _build_parser()
    args = parser.parse_args(argv)

    classifier = _build_classifier(args)
    config = SplitterConfig(
        output_dir=Path(args.output),
        batch_size=args.batch_size,
        max_stored_text_chars=args.max_stored_text_chars,
        use_pdfplumber_fallback=not args.disable_pdfplumber,
        render_dpi=args.render_dpi,
        image_format=args.image_format,
        image_quality=args.image_quality,
        keep_page_images=args.keep_page_images,
    )

    try:
        splitter = ClaimFileSplitter(classifier=classifier, config=config)
        result = splitter.run(args.input_pdf)
        summary = {
            "source_pdf": str(result.source_pdf),
            "output_dir": str(result.output_dir),
            "manifest_path": str(result.manifest_path),
            "page_count": len(result.pages),
            "document_count": len(result.segments),
            "documents": [
                {
                    "document_type": written.segment.document_type,
                    "start_page": written.segment.start_page,
                    "end_page": written.segment.end_page,
                    "output_path": str(written.output_path),
                }
                for written in result.written_documents
            ],
        }
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        close = getattr(classifier, "close", None)
        if callable(close):
            close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split and classify a large insurance claim-file PDF.",
    )
    parser.add_argument("input_pdf", type=Path, help="Path to the claim-file PDF.")
    parser.add_argument(
        "--output",
        default="output",
        help="Directory for split PDFs and manifest.json.",
    )
    parser.add_argument(
        "--classifier",
        choices=("auto", "azure", "rules"),
        default="auto",
        help="Classifier backend. auto uses Azure when configuration is present.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help=_env_file_help(),
    )
    parser.add_argument(
        "--project-endpoint",
        default=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        help="Azure AI Foundry project endpoint. Defaults to AZURE_AI_PROJECT_ENDPOINT.",
    )
    parser.add_argument(
        "--deployment",
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        help="Azure OpenAI model deployment name. Defaults to AZURE_OPENAI_DEPLOYMENT.",
    )
    parser.add_argument(
        "--api",
        choices=("responses", "chat"),
        default="responses",
        help="Azure OpenAI API surface used by the OpenAI client.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of pages sent to the classifier per model call.",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=160,
        help="DPI used when rendering PDF pages to images for Azure classification.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpeg", "png"),
        default="jpeg",
        help="Rendered page image format for Azure classification.",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=85,
        help="JPEG quality used for rendered page images.",
    )
    parser.add_argument(
        "--keep-page-images",
        action="store_true",
        help="Keep rendered page images under output/page_images for inspection.",
    )
    parser.add_argument(
        "--max-stored-text-chars",
        type=int,
        default=12000,
        help="Maximum extracted text retained per page before classification.",
    )
    parser.add_argument(
        "--disable-pdfplumber",
        action="store_true",
        help="Disable pdfplumber text fallback and use pypdf extraction only.",
    )
    return parser


def load_cli_environment(argv: list[str] | None = None) -> None:
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", type=Path)
    known_args, _ = env_parser.parse_known_args(argv)
    load_dotenv(dotenv_path=known_args.env_file, override=False)


def _env_file_help() -> str:
    return (
        "Path to a dotenv file. Defaults to python-dotenv's standard .env "
        "discovery from the current working directory."
    )


def _build_classifier(args: argparse.Namespace):
    wants_azure = args.classifier == "azure"
    can_use_azure = bool(args.project_endpoint and args.deployment)

    if args.classifier == "rules" or (args.classifier == "auto" and not can_use_azure):
        if args.classifier == "auto" and not can_use_azure:
            print(
                "Azure configuration not found; using rule-based classifier. "
                "Set AZURE_AI_PROJECT_ENDPOINT and AZURE_OPENAI_DEPLOYMENT to use Azure.",
                file=sys.stderr,
            )
        return RuleBasedPageClassifier()

    if wants_azure and not can_use_azure:
        raise SystemExit(
            "--classifier azure requires --project-endpoint and --deployment "
            "or AZURE_AI_PROJECT_ENDPOINT and AZURE_OPENAI_DEPLOYMENT."
        )

    return AzureProjectPageClassifier(
        project_endpoint=args.project_endpoint,
        deployment=args.deployment,
        use_responses_api=args.api == "responses",
    )
