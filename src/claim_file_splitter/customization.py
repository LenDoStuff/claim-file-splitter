from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import create_model


DEFAULT_CATEGORIES = [
    {
        "name": "repair_invoices",
        "filename_prefix": "repair_invoice",
        "description": "Repair invoices, body shop bills, parts, labor, and amount due pages.",
    },
    {
        "name": "appraisals",
        "filename_prefix": "appraisal",
        "description": "Appraisals, valuation reports, and damage estimates.",
    },
    {
        "name": "communications",
        "filename_prefix": "communication",
        "description": "Emails, letters, claim notes, and general communications.",
    },
    {
        "name": "police_reports",
        "filename_prefix": "police_report",
        "description": "Police, incident, crash, and officer reports.",
    },
    {
        "name": "photos",
        "filename_prefix": "photo_section",
        "description": "Photos, photo logs, and image-only damage sections.",
    },
    {
        "name": "payments",
        "filename_prefix": "payment_document",
        "description": "Payment notices, checks, remittances, and settlement payments.",
    },
    {
        "name": "medical",
        "filename_prefix": "medical_document",
        "description": "Medical records, treatment notes, and provider documents.",
    },
    {
        "name": "legal_correspondence",
        "filename_prefix": "legal_correspondence",
        "description": "Attorney letters, legal demands, and litigation correspondence.",
    },
    {
        "name": "other",
        "filename_prefix": "document",
        "description": "Fallback category for pages that do not match configured categories.",
    },
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a claim-file document boundary detector and classifier. "
    "Return only structured data."
)

DEFAULT_USER_PROMPT = (
    "Classify the attached target page images from one insurance claim file. "
    "Use rolling context only to decide whether the first page continues an "
    "already open document. Classify only the target pages listed in the text "
    "metadata and attached as images. Do not classify pages that appear only "
    "inside rolling_context."
)


class AzureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_endpoint: str | None = None
    deployment: str | None = None
    temperature: float = 0.0


class CategoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    filename_prefix: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
            raise ValueError(
                "category names must use lowercase letters, numbers, and underscores "
                "and must start with a letter"
            )
        return value

    @field_validator("filename_prefix")
    @classmethod
    def validate_filename_prefix(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
            raise ValueError(
                "filename prefixes must use lowercase letters, numbers, and underscores "
                "and must start with a letter"
            )
        return value


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT


class SplitterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=5, ge=1)
    recent_page_decision_limit: int = Field(default=5, ge=0)
    completed_document_limit: int = Field(default=3, ge=0)
    high_confidence_batch_boundary: float = Field(default=0.75, ge=0.0, le=1.0)
    other_type_boundary_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    type_change_boundary_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_stored_text_chars: int = Field(default=12000, ge=0)


class RenderingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dpi: int = Field(default=160, ge=72)
    image_format: str = "jpeg"
    image_quality: int = Field(default=85, ge=1, le=100)
    image_detail: str = "high"
    keep_page_images: bool = False

    @field_validator("image_format")
    @classmethod
    def validate_image_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"jpeg", "jpg", "png"}:
            raise ValueError("image_format must be 'jpeg' or 'png'")
        return "jpeg" if normalized == "jpg" else normalized


class ClaimSplitterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    azure: AzureConfig = Field(default_factory=AzureConfig)
    categories: list[CategoryConfig] = Field(
        default_factory=lambda: [CategoryConfig(**item) for item in DEFAULT_CATEGORIES]
    )
    default_document_type: str = "other"
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)

    @model_validator(mode="after")
    def validate_categories(self) -> "ClaimSplitterConfig":
        names = [category.name for category in self.categories]
        if not names:
            raise ValueError("at least one category is required")
        if len(set(names)) != len(names):
            raise ValueError("category names must be unique")
        if self.default_document_type not in names:
            raise ValueError("default_document_type must match a configured category")
        return self


class _PageDecisionOutputBase(BaseModel):
    page: int = Field(description="The one-based source PDF page number.")
    starts_new_document: bool = Field(
        description="True when this page starts a new logical document."
    )
    title: str = Field(description="Short visible or inferred document title.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence from 0 to 1 for this page decision.",
    )
    reason: str = Field(description="Short reason for the boundary/type decision.")


def resolve_config(
    *,
    categories: list[CategoryConfig | dict[str, Any]] | None = None,
    default_document_type: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    project_endpoint: str | None = None,
    deployment: str | None = None,
    temperature: float | None = None,
    batch_size: int | None = None,
    recent_page_decision_limit: int | None = None,
    completed_document_limit: int | None = None,
    high_confidence_batch_boundary: float | None = None,
    other_type_boundary_confidence: float | None = None,
    type_change_boundary_confidence: float | None = None,
    render_dpi: int | None = None,
    image_format: str | None = None,
    image_quality: int | None = None,
    image_detail: str | None = None,
    keep_page_images: bool | None = None,
    max_stored_text_chars: int | None = None,
) -> ClaimSplitterConfig:
    payload = ClaimSplitterConfig().model_dump(mode="python")
    if categories is not None:
        payload["categories"] = categories
    if default_document_type is not None:
        payload["default_document_type"] = default_document_type
    if system_prompt is not None:
        payload["prompts"]["system_prompt"] = system_prompt
    if user_prompt is not None:
        payload["prompts"]["user_prompt"] = user_prompt
    if project_endpoint is not None:
        payload["azure"]["project_endpoint"] = project_endpoint
    if deployment is not None:
        payload["azure"]["deployment"] = deployment
    if temperature is not None:
        payload["azure"]["temperature"] = temperature
    if batch_size is not None:
        payload["splitter"]["batch_size"] = batch_size
    if recent_page_decision_limit is not None:
        payload["splitter"]["recent_page_decision_limit"] = recent_page_decision_limit
    if completed_document_limit is not None:
        payload["splitter"]["completed_document_limit"] = completed_document_limit
    if high_confidence_batch_boundary is not None:
        payload["splitter"]["high_confidence_batch_boundary"] = (
            high_confidence_batch_boundary
        )
    if other_type_boundary_confidence is not None:
        payload["splitter"]["other_type_boundary_confidence"] = (
            other_type_boundary_confidence
        )
    if type_change_boundary_confidence is not None:
        payload["splitter"]["type_change_boundary_confidence"] = (
            type_change_boundary_confidence
        )
    if max_stored_text_chars is not None:
        payload["splitter"]["max_stored_text_chars"] = max_stored_text_chars
    if render_dpi is not None:
        payload["rendering"]["dpi"] = render_dpi
    if image_format is not None:
        payload["rendering"]["image_format"] = image_format
    if image_quality is not None:
        payload["rendering"]["image_quality"] = image_quality
    if image_detail is not None:
        payload["rendering"]["image_detail"] = image_detail
    if keep_page_images is not None:
        payload["rendering"]["keep_page_images"] = keep_page_images
    return ClaimSplitterConfig.model_validate(payload)


def make_batch_classification_output_model(config: ClaimSplitterConfig) -> type[BaseModel]:
    document_type = Enum(
        "DocumentType",
        {category.name: category.name for category in config.categories},
        type=str,
    )
    page_decision_output = create_model(
        "PageDecisionOutput",
        __base__=_PageDecisionOutputBase,
        document_type=(
            document_type,
            Field(description="The configured document category for this page."),
        ),
    )
    return create_model(
        "BatchClassificationOutput",
        pages=(list[page_decision_output], Field(default_factory=list)),
    )
