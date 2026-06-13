# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

This is a Python package for splitting one large insurance claim-file PDF into
logical documents, classifying each document, and writing split PDFs into
document-type folders.

The primary public API is `split_claim_file_azure(...)`. It uses Azure AI
Projects to obtain an Azure OpenAI client and classifies rendered PDF page
images with structured output:

```python
client.responses.parse(..., text_format=make_batch_classification_output_model(config))
```

The rule-based classifier is kept for local smoke runs and tests without Azure
credentials; it is not the main library API.

## Important Files

- `src/claim_file_splitter/customization.py`
  - Pydantic config schema, JSON config loading, built-in defaults, and dynamic
    structured-output model generation.
- `src/claim_file_splitter/models.py`
  - Pydantic public result models and small internal conversion helpers.
- `src/claim_file_splitter/classifiers.py`
  - Azure structured-output classifier and local rule-based classifier.
- `src/claim_file_splitter/pipeline.py`
  - Page batching, rolling context, boundary reconciliation, segment building,
    and orchestration.
- `src/claim_file_splitter/pdf.py`
  - PDF analysis, page rendering, and split-PDF writing.
- `src/claim_file_splitter/cli.py`
  - CLI entry point and dotenv loading.
- `tests/`
  - Unit tests for CLI env loading, rendering, Azure request shape, batching,
    segmentation, and PDF output behavior.

## Development Commands

Install for development:

```powershell
python -m pip install -e ".[dev]"
```

Run the test suite:

```powershell
python -m pytest -q
```

Compile-check source and tests:

```powershell
python -m compileall -q src tests
```

Check CLI help:

```powershell
python -m claim_file_splitter --help
```

Local rule-based smoke run:

```powershell
claim-file-splitter .\claim-file.pdf --output .\output --classifier rules
```

Azure run:

```powershell
claim-file-splitter .\claim-file.pdf --output .\output --config .\splitter.json
```

Public API smoke:

```python
from claim_file_splitter import split_claim_file_azure

result = split_claim_file_azure("claim-file.pdf", output_dir="output")
```

## Azure Classifier Requirements

- The Azure classifier must use rendered page images as the model input.
- Keep the default batch size at five pages unless intentionally changing the
  active `ClaimSplitterConfig`.
- Include rolling context in prompts only for continuity across batch breaks.
- Do not embed extracted PDF page text in the Azure prompt payload.
- Keep each image item shaped as an Responses API `input_image` with
  `detail` set from `config.rendering.image_detail`.
- Keep structured output typed through the dynamic
  `make_batch_classification_output_model(config)` result; do not reintroduce
  regex JSON extraction or prompt-level response-shape parsing.

## PDF Output Requirements

- Each detected logical document must be written as one PDF, even when it spans
  multiple source pages.
- Split PDFs must copy pages from the original source PDF rather than rebuilding
  pages from extracted text.
- Output folders and filename prefixes come from
  `ClaimSplitterConfig.categories`.

## Testing Expectations

When changing classifier, batching, segmentation, rendering, or CLI behavior,
run:

```powershell
python -m pytest -q
python -m compileall -q src tests
python -m claim_file_splitter --help
```

For Azure classifier changes, keep or update tests that prove:

- `responses.parse` is used.
- `text_format` is the dynamic model generated from the active config.
- image items include the configured image detail value.
- embedded PDF text is not present in the Azure prompt payload.
- multi-page documents are emitted as one output PDF with the expected original
  pages.
- JSON config can replace categories and filename prefixes.
- direct function/CLI args override config file and environment values.

## Repository Hygiene

- Do not commit secrets or local `.env` files.
- Keep changes scoped to the requested behavior.
- Do not push or create pull requests unless explicitly asked.
- Preserve the function-oriented style unless a broader refactor is explicitly
  requested.

## Code Style

- Prefer simple functions, clear data flow, and minimal abstractions.
- Avoid unnecessary classes, wrappers, factories, interfaces, and defensive
  programming unless they solve a concrete problem.
- Keep the code easy to read and easy to modify.
