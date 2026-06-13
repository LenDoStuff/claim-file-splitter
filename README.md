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

Configuration precedence is:

```text
direct function/CLI args > Python config object > JSON config file > environment variables > built-in defaults
```

Environment variables:

```powershell
$env:AZURE_AI_PROJECT_ENDPOINT="https://YOUR-RESOURCE-NAME.services.ai.azure.com/api/projects/YOUR-PROJECT-NAME"
$env:AZURE_OPENAI_DEPLOYMENT="gpt-4.1-mini"
```

The CLI also loads `.env` from the current working directory, or a specific file
with `--env-file`.

## JSON Config

Use JSON to configure categories, prompts, Azure settings, splitter settings,
and image rendering:

```json
{
  "azure": {
    "project_endpoint": "https://YOUR-RESOURCE-NAME.services.ai.azure.com/api/projects/YOUR-PROJECT-NAME",
    "deployment": "gpt-4.1-mini",
    "temperature": 0
  },
  "categories": [
    {
      "name": "repair_invoices",
      "filename_prefix": "repair_invoice",
      "description": "Repair invoices and body shop bills.",
      "rule_keywords": ["repair invoice", "body shop", "amount due"]
    },
    {
      "name": "other",
      "filename_prefix": "document",
      "description": "Fallback category.",
      "rule_keywords": []
    }
  ],
  "default_document_type": "other",
  "prompts": {
    "system_prompt": "You are a claim-file document boundary detector and classifier. Return only structured data.",
    "user_prompt": "Classify only the attached target page images. Use rolling context only for continuity."
  },
  "splitter": {
    "batch_size": 5,
    "recent_page_decision_limit": 5,
    "completed_document_limit": 3,
    "high_confidence_batch_boundary": 0.75,
    "other_type_boundary_confidence": 0.75,
    "type_change_boundary_confidence": 0.5,
    "max_stored_text_chars": 12000,
    "use_pdfplumber_fallback": true
  },
  "rendering": {
    "dpi": 160,
    "image_format": "jpeg",
    "image_quality": 85,
    "image_detail": "high",
    "keep_page_images": false
  }
}
```

If `categories` is provided, it replaces the built-in categories completely.
Configured category names drive both output folders and the Azure structured
output enum.

## CLI

Run with Azure:

```powershell
claim-file-splitter .\claim-file.pdf --output .\output --config .\splitter.json
```

Override common config values from the CLI:

```powershell
claim-file-splitter .\claim-file.pdf `
  --config .\splitter.json `
  --deployment gpt-4.1-mini `
  --batch-size 5 `
  --render-dpi 160 `
  --keep-page-images
```

Local rule-based smoke run without Azure:

```powershell
claim-file-splitter .\claim-file.pdf --output .\output --classifier rules
```

## Output

The module writes one PDF per detected logical document and a manifest:

```text
output/
  repair_invoices/
    repair_invoice_001.pdf
  appraisals/
    appraisal_001.pdf
  communications/
    communication_001.pdf
  manifest.json
```

The folder names and filename prefixes come from the active config categories.

## How It Works

1. Reads the source PDF page by page and keeps embedded text for local smoke runs.
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
python -m claim_file_splitter --help
```
