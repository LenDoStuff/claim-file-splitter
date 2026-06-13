# Claim File Splitter

Python module for splitting a large insurance claim-file PDF into logical
documents, classifying each extracted document with Azure OpenAI structured
output, and writing split PDFs into document-type folders.

The primary library API is Azure-first:

```python
from claim_file_splitter import split_claim_file_azure

result = split_claim_file_azure(
    "claim-file.pdf",
    output_dir="output",
    project_endpoint="https://YOUR-RESOURCE-NAME.services.ai.azure.com/api/projects/YOUR-PROJECT-NAME",
    deployment="gpt-4.1-mini",
)

for document in result.documents:
    print(document.segment.document_type, document.output_path)
```

`split_claim_file_azure(...)` returns a typed `ClaimSplitResult` Pydantic model.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Azure Configuration

Authenticate with Azure first:

```powershell
az login
```

Configuration precedence is intentionally simple:

```text
direct function args > built-in defaults
```

The typed defaults are visible in Python:

```python
from claim_file_splitter import ClaimSplitterConfig

defaults = ClaimSplitterConfig()
print(defaults.model_dump())
```

Runtime customization is done with explicit arguments:

```python
result = split_claim_file_azure(
    "claim-file.pdf",
    output_dir="output",
    project_endpoint="https://YOUR-RESOURCE-NAME.services.ai.azure.com/api/projects/YOUR-PROJECT-NAME",
    deployment="gpt-4.1-mini",
    temperature=0,
    batch_size=5,
    categories=[
        {
            "name": "repair_invoices",
            "filename_prefix": "repair_invoice",
            "description": "Repair invoices and body shop bills.",
        },
        {
            "name": "other",
            "filename_prefix": "document",
            "description": "Fallback category.",
        },
    ],
    default_document_type="other",
    system_prompt="You are a claim-file document boundary detector and classifier. Return only structured data.",
    user_prompt="Classify only the attached target page images. Use rolling context only for continuity.",
)
```

When `categories` is passed, it replaces the built-in categories completely.
Category names drive output folders and the Azure structured-output enum.

### Config Reference

Config values resolve in this order:

```text
direct overrides > ClaimSplitterConfig defaults
```

`azure` controls the Azure client and model call:

| Field | What it does |
| --- | --- |
| `project_endpoint` | Azure AI Foundry project endpoint passed to `AIProjectClient`. Pass with `project_endpoint=...`. |
| `deployment` | Azure OpenAI deployment name used as the `model` value. Pass with `deployment=...`. |
| `temperature` | Temperature passed to `client.responses.parse(...)`. Pass with `temperature=...`. Default is `0` for stable classification. |

`categories` controls document types, output folders, filenames, and structured output:

| Field | What it does |
| --- | --- |
| `name` | Document type identifier. It must use lowercase letters, numbers, and underscores, starting with a letter. This becomes the output folder name and an allowed structured-output enum value. |
| `filename_prefix` | Prefix for split PDFs inside that category folder, for example `repair_invoice_001.pdf`. |
| `description` | Human-readable category guidance sent to the model in the prompt payload. |

`default_document_type` is the fallback category name used when a classifier
decision is missing or invalid. It must match one configured category.

`prompts` controls the model instructions:

| Field | What it does |
| --- | --- |
| `system_prompt` | System message sent to the Azure OpenAI Responses API. |
| `user_prompt` | Instruction text included in the user payload with batch metadata and rendered page images. |

`splitter` controls batching, context, and boundary reconciliation:

All splitter fields can be passed as Python direct overrides with the same
names shown below.

| Field | What it does |
| --- | --- |
| `batch_size` | Number of PDF pages rendered and sent per model call. Default is `5`. |
| `recent_page_decision_limit` | Number of recent page decisions included in rolling context for the next batch. |
| `completed_document_limit` | Number of completed document summaries included in rolling context. |
| `high_confidence_batch_boundary` | Minimum confidence needed to preserve a new-document boundary on the first page of a later batch. Lower-confidence first-page boundaries are treated as continuation. |
| `other_type_boundary_confidence` | Confidence needed to force a new segment when one side of the type change is the default document type. |
| `type_change_boundary_confidence` | Confidence needed to force a new segment when two non-default document types differ. |
| `max_stored_text_chars` | Maximum extracted text characters retained internally per page for metadata/debugging. Azure classification still uses rendered page images. |

`rendering` controls page image generation:

| Field | What it does |
| --- | --- |
| `dpi` | DPI used to render PDF pages into images before Azure classification. Pass as `render_dpi=...`. |
| `image_format` | Rendered image format: `jpeg` or `png`. Pass with `image_format=...`. |
| `image_quality` | JPEG quality from `1` to `100`; only used for JPEG output. Pass with `image_quality=...`. |
| `image_detail` | Responses API image detail value, usually `high`. Pass with `image_detail=...`. |
| `keep_page_images` | When `true`, rendered images are saved under `output/page_images/`; otherwise they are temporary and removed after the run. Pass with `keep_page_images=True`. |

## Output

The module writes one PDF per detected logical document and one manifest file.
Only folders for document types that are actually detected are created.

For this configured category:

```json
{
  "name": "repair_invoices",
  "filename_prefix": "repair_invoice",
  "description": "Repair invoices and body shop bills."
}
```

an extracted repair invoice becomes:

```text
output/
  repair_invoices/
    repair_invoice_001.pdf
```

The exact naming rule is:

```text
output/{document_type}/{filename_prefix}_{counter:03d}.pdf
```

Counters are independent per document type. If the pipeline finds two repair
invoices and one payment document, expect:

```text
output/
  repair_invoices/
    repair_invoice_001.pdf
    repair_invoice_002.pdf
  payments/
    payment_document_001.pdf
  manifest.json
```

Each split PDF contains the original source PDF pages for that segment. A
3-page invoice is written as one 3-page PDF, not as three separate PDFs.

When `rendering.keep_page_images` is `true`, the output also includes rendered
page images:

```text
output/
  page_images/
    page_000001.jpg
    page_000002.jpg
  manifest.json
```

These images are for inspection/debugging. They are not required to use the
split PDFs.

### Manifest Structure

`manifest.json` is written to the output directory and has this top-level shape:

```json
{
  "source_pdf": "claim-file.pdf",
  "output_dir": "output",
  "page_count": 12,
  "document_count": 3,
  "pages": [],
  "page_decisions": [],
  "classification_batches": [],
  "documents": []
}
```

`pages` contains one entry per source PDF page:

```json
{
  "page_number": 1,
  "word_count": 120,
  "char_count": 840,
  "image_count": 2,
  "is_image_only": false,
  "may_require_ocr": false,
  "rendered_image": {
    "page_number": 1,
    "mime_type": "image/jpeg",
    "width_px": 1224,
    "height_px": 1584,
    "byte_size": 184321,
    "path": "output/page_images/page_000001.jpg"
  }
}
```

`rendered_image` appears only when the page was rendered for Azure
classification. `path` appears only when `keep_page_images` is `true`. The image
data URI sent to Azure is not written to the manifest.

`page_decisions` contains one classifier decision per source PDF page:

```json
{
  "page_number": 1,
  "document_type": "repair_invoices",
  "starts_new_document": true,
  "title": "Repair Invoice",
  "confidence": 0.92,
  "reason": "Visible invoice header and line-item repair details."
}
```

`classification_batches` records model-call batching and continuity context:

```json
{
  "batch_number": 1,
  "start_page": 1,
  "end_page": 5,
  "page_numbers": [1, 2, 3, 4, 5],
  "rolling_context": {
    "open_document": null,
    "recent_page_decisions": [],
    "completed_documents": [],
    "document_type_counts": {}
  },
  "reconciliation_messages": []
}
```

Later batches include the prior open document, recent decisions, completed
document summaries, and document type counts in `rolling_context`.

`documents` contains one entry per written split PDF:

```json
{
  "segment_id": 1,
  "document_type": "repair_invoices",
  "start_page": 1,
  "end_page": 3,
  "page_count": 3,
  "title": "Repair Invoice",
  "summary": "Repair Invoice. Visible invoice header and continuation pages share invoice details.",
  "confidence": 0.91,
  "reasons": [
    "Visible invoice header.",
    "Continuation pages share invoice details."
  ],
  "output_path": "output/repair_invoices/repair_invoice_001.pdf"
}
```

`start_page` and `end_page` are one-based, inclusive source PDF page numbers.
`summary` is a short segment-level description derived from page titles and
classifier reasons. `confidence` is the average confidence across the page
decisions in that document segment.

## How It Works

1. Reads the source PDF page by page and collects page metadata.
2. Renders Azure-classified PDF pages to images locally.
3. Sends batches of target page images to the Azure OpenAI client from
   `AIProjectClient.get_openai_client()`.
4. Uses `client.responses.parse(..., text_format=...)` with a dynamic structured
   output model generated from configured categories.
5. Carries rolling context between batches so documents can continue across
   batch breaks.
6. Splits original source PDF pages with `pypdf`, preserving original pages.
7. Writes split PDFs and `manifest.json`.

The Azure prompt path treats rendered images as the authoritative page input. It
does not send embedded PDF text excerpts to the model.

## Development Checks

```powershell
python -m pytest -q
python -m compileall -q src tests
```
