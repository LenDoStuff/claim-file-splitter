# Claim File Splitter

Python pipeline for splitting a large insurance claim-file PDF into logical documents, classifying each extracted document, and writing the split PDFs into document-type folders.

The production path uses Azure AI Projects to create an authenticated Azure OpenAI client. A deterministic rule-based classifier is included for local smoke tests and CI without Azure credentials.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Azure configuration

Authenticate with Azure first:

```powershell
az login
```

Set the Foundry project endpoint and the model deployment name:

```powershell
$env:AZURE_AI_PROJECT_ENDPOINT="https://YOUR-RESOURCE-NAME.services.ai.azure.com/api/projects/YOUR-PROJECT-NAME"
$env:AZURE_OPENAI_DEPLOYMENT="gpt-4.1-mini"
```

Run the splitter:

```powershell
claim-file-splitter .\claim-file.pdf --output .\output --classifier azure
```

For a local dry run without Azure:

```powershell
claim-file-splitter .\claim-file.pdf --output .\output --classifier rules
```

## Output

The pipeline writes one PDF per detected logical document and a machine-readable manifest:

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

Supported document-type folders:

- `repair_invoices`
- `appraisals`
- `communications`
- `police_reports`
- `photos`
- `payments`
- `medical`
- `legal_correspondence`
- `other`

## How it works

1. Reads the source PDF page by page with embedded text extraction first.
2. Captures page-level signals such as word count, image count, text excerpt, and image-only pages that may require OCR later.
3. Sends batches of page summaries to the Azure OpenAI client obtained from `AIProjectClient.get_openai_client()`.
4. Receives page-level boundary and document-type decisions.
5. Consolidates contiguous pages into logical document segments.
6. Splits the original PDF pages with `pypdf`, preserving the original page rendering.
7. Writes split PDFs under folders named by document type and saves `manifest.json`.

The first version does not perform OCR by default. It avoids depending on OCR for normal text PDFs and flags image-only pages in the manifest so an OCR provider can be added only where needed.

## Relevant Microsoft docs

- Azure AI Projects Python SDK: https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme
- `AIProjectClient.get_openai_client()`: https://learn.microsoft.com/en-us/python/api/azure-ai-projects/azure.ai.projects.aiprojectclient
- Azure OpenAI v1 API with the `OpenAI()` client: https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/how-to/use-chat-completions
