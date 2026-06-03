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

Or create a local `.env` file from `.env.example`:

```text
AZURE_AI_PROJECT_ENDPOINT=https://YOUR-RESOURCE-NAME.services.ai.azure.com/api/projects/YOUR-PROJECT-NAME
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
```

The CLI loads `.env` from the current working directory before reading defaults. To use a file somewhere else:

```powershell
claim-file-splitter .\claim-file.pdf --env-file .\path\to\.env --classifier azure
```

Use a vision-capable Azure OpenAI deployment. The Azure classifier renders PDF pages locally and sends those page images to the model.

Run the splitter:

```powershell
claim-file-splitter .\claim-file.pdf --output .\output --classifier azure
```

For a local dry run without Azure:

```powershell
claim-file-splitter .\claim-file.pdf --output .\output --classifier rules
```

For Python PoC code, call the function directly:

```python
from claim_file_splitter import split_claim_file
from claim_file_splitter.classifiers import rule_based_classify_pages

result = split_claim_file(
    "claim-file.pdf",
    output_dir="output",
    classify_pages=rule_based_classify_pages,
)
```

## Customization

Project-specific prompts, document categories, output filename prefixes, image detail, batch size, and the structured output schema live in:

```text
src/claim_file_splitter/customization.py
```

Edit that file when adapting the PoC to a different claim workflow. The Azure path uses `client.responses.parse(..., text_format=BatchClassificationOutput)` so the model returns typed page decisions instead of free-form JSON.

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

1. Reads the source PDF page by page and keeps embedded text available for local rule-based dry runs.
2. Renders each Azure-classified PDF page to an image locally.
3. Sends batches of five target page images to the Azure OpenAI client obtained from `AIProjectClient.get_openai_client()`.
4. Receives page-level boundary and document-type decisions.
5. Carries rolling context between batches so documents can continue across pages `5/6`, `10/11`, and later breaks.
6. Splits the original PDF pages with `pypdf`, preserving the original page rendering.
7. Writes split PDFs under folders named by document type and saves `manifest.json`.

The Azure prompt path treats rendered images as the authoritative page input. It does not send embedded PDF text excerpts to the model. The local rule-based classifier still uses embedded text so tests and smoke runs work without Azure credentials.

## Batch and rendering options

The default Azure batch size is five pages:

```powershell
claim-file-splitter .\claim-file.pdf --classifier azure --batch-size 5
```

Useful rendering controls:

```powershell
claim-file-splitter .\claim-file.pdf `
  --classifier azure `
  --render-dpi 160 `
  --image-format jpeg `
  --image-quality 85 `
  --keep-page-images
```

When `--keep-page-images` is set, rendered images are saved under `output/page_images/` for inspection. Otherwise they are created in a temporary directory and removed after classification.

## Relevant Microsoft docs

- Azure AI Projects Python SDK: https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme
- `AIProjectClient.get_openai_client()`: https://learn.microsoft.com/en-us/python/api/azure-ai-projects/azure.ai.projects.aiprojectclient
- Azure OpenAI v1 API with the `OpenAI()` client: https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/how-to/use-chat-completions
- Azure OpenAI image inputs: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/gpt-with-vision
