from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import split_claim_file_azure


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    load_cli_environment(argv)
    args = build_parser().parse_args(argv)

    common_kwargs = {
        "output_dir": args.output,
        "config_path": args.config,
        "batch_size": args.batch_size,
        "render_dpi": args.render_dpi,
        "image_format": args.image_format,
        "image_quality": args.image_quality,
        "keep_page_images": True if args.keep_page_images else None,
        "max_stored_text_chars": args.max_stored_text_chars,
    }

    result = split_claim_file_azure(
        args.input_pdf,
        project_endpoint=args.project_endpoint,
        deployment=args.deployment,
        image_detail=args.image_detail,
        **common_kwargs,
    )

    print(json.dumps(cli_summary(result), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
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
        "--config",
        type=Path,
        help="Path to a JSON splitter configuration file.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help=(
            "Path to a dotenv file. Defaults to python-dotenv's standard .env "
            "discovery from the current working directory."
        ),
    )
    parser.add_argument(
        "--project-endpoint",
        help="Azure AI Foundry project endpoint. Overrides config and env.",
    )
    parser.add_argument(
        "--deployment",
        help="Azure OpenAI model deployment name. Overrides config and env.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of pages sent to the classifier per model call.",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        help="DPI used when rendering PDF pages to images for Azure classification.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpeg", "png"),
        help="Rendered page image format for Azure classification.",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        help="JPEG quality used for rendered page images.",
    )
    parser.add_argument(
        "--image-detail",
        help="Responses API image detail value. Defaults to config rendering.image_detail.",
    )
    parser.add_argument(
        "--keep-page-images",
        action="store_true",
        help="Keep rendered page images under output/page_images for inspection.",
    )
    parser.add_argument(
        "--max-stored-text-chars",
        type=int,
        help="Maximum extracted text retained per page before classification.",
    )
    return parser


def load_cli_environment(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", type=Path)
    known_args, _ = parser.parse_known_args(argv)
    load_dotenv(dotenv_path=known_args.env_file, override=False)


def cli_summary(result) -> dict:
    return {
        "source_pdf": str(result.source_pdf),
        "output_dir": str(result.output_dir),
        "manifest_path": str(result.manifest_path),
        "page_count": result.page_count,
        "document_count": result.document_count,
        "documents": [
            {
                "document_type": document.segment.document_type,
                "start_page": document.segment.start_page,
                "end_page": document.segment.end_page,
                "summary": document.segment.summary,
                "output_path": str(document.output_path),
            }
            for document in result.documents
        ],
    }
