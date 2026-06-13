from __future__ import annotations

import json
import os
import re
from enum import Enum
from pathlib import Path
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


def load_config_dict(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must contain an object at the top level.")
    return payload


def resolve_config(
    *,
    config: ClaimSplitterConfig | dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    project_endpoint: str | None = None,
    deployment: str | None = None,
    batch_size: int | None = None,
    render_dpi: int | None = None,
    image_format: str | None = None,
    image_quality: int | None = None,
    image_detail: str | None = None,
    keep_page_images: bool | None = None,
    max_stored_text_chars: int | None = None,
) -> ClaimSplitterConfig:
    payload = ClaimSplitterConfig().model_dump(mode="python")
    merge_config(payload, env_config_dict())
    if config_path is not None:
        merge_config(payload, load_config_dict(config_path))
    if config is not None:
        merge_config(payload, config_override_dict(config))
    merge_config(
        payload,
        direct_override_dict(
            project_endpoint=project_endpoint,
            deployment=deployment,
            batch_size=batch_size,
            render_dpi=render_dpi,
            image_format=image_format,
            image_quality=image_quality,
            image_detail=image_detail,
            keep_page_images=keep_page_images,
            max_stored_text_chars=max_stored_text_chars,
        ),
    )
    return ClaimSplitterConfig.model_validate(payload)


def env_config_dict() -> dict[str, Any]:
    azure = {}
    if os.getenv("AZURE_AI_PROJECT_ENDPOINT"):
        azure["project_endpoint"] = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if os.getenv("AZURE_OPENAI_DEPLOYMENT"):
        azure["deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    return {"azure": azure} if azure else {}


def config_override_dict(config: ClaimSplitterConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, ClaimSplitterConfig):
        return config.model_dump(mode="python", exclude_unset=True)
    if isinstance(config, dict):
        return config
    raise TypeError("config must be a ClaimSplitterConfig or dict.")


def direct_override_dict(
    *,
    project_endpoint: str | None,
    deployment: str | None,
    batch_size: int | None,
    render_dpi: int | None,
    image_format: str | None,
    image_quality: int | None,
    image_detail: str | None,
    keep_page_images: bool | None,
    max_stored_text_chars: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"azure": {}, "splitter": {}, "rendering": {}}
    if project_endpoint is not None:
        payload["azure"]["project_endpoint"] = project_endpoint
    if deployment is not None:
        payload["azure"]["deployment"] = deployment
    if batch_size is not None:
        payload["splitter"]["batch_size"] = batch_size
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
    return {
        key: value
        for key, value in payload.items()
        if not isinstance(value, dict) or value
    }


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
            and key != "categories"
        ):
            merge_config(base[key], value)
        else:
            base[key] = value


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
