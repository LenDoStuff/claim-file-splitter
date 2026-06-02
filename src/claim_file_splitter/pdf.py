from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from .models import DOCUMENT_TYPE_PREFIXES, DocumentSegment, PageFeatures, WrittenDocument


def analyze_pdf(
    input_pdf: str | Path,
    *,
    max_stored_text_chars: int = 12000,
    use_pdfplumber_fallback: bool = True,
) -> list[PageFeatures]:
    path = Path(input_pdf)
    reader = PdfReader(str(path))
    fallback_texts = (
        _extract_with_pdfplumber(path) if use_pdfplumber_fallback else {}
    )

    pages: list[PageFeatures] = []
    for index, page in enumerate(reader.pages):
        page_number = index + 1
        text = page.extract_text() or fallback_texts.get(index, "") or ""
        text = _clean_text(text)
        if len(text) > max_stored_text_chars:
            text = text[:max_stored_text_chars].rstrip()
        image_count = _count_page_images(page)
        word_count = len(re.findall(r"\w+", text))
        char_count = len(text)
        is_image_only = word_count == 0 and image_count > 0
        pages.append(
            PageFeatures(
                source_path=path,
                page_number=page_number,
                text=text,
                word_count=word_count,
                char_count=char_count,
                image_count=image_count,
                is_image_only=is_image_only,
                may_require_ocr=is_image_only,
            )
        )
    return pages


def split_pdf(
    input_pdf: str | Path,
    segments: Iterable[DocumentSegment],
    output_dir: str | Path,
) -> list[WrittenDocument]:
    source = Path(input_pdf)
    root = Path(output_dir)
    reader = PdfReader(str(source))
    written: list[WrittenDocument] = []
    counters: dict[str, int] = {}

    for segment in segments:
        target_dir = root / segment.document_type
        target_dir.mkdir(parents=True, exist_ok=True)
        counters[segment.document_type] = counters.get(segment.document_type, 0) + 1
        prefix = DOCUMENT_TYPE_PREFIXES[segment.document_type]
        output_path = target_dir / f"{prefix}_{counters[segment.document_type]:03d}.pdf"

        writer = PdfWriter()
        for page_number in range(segment.start_page, segment.end_page + 1):
            writer.add_page(reader.pages[page_number - 1])
        with output_path.open("wb") as handle:
            writer.write(handle)
        written.append(WrittenDocument(segment=segment, output_path=output_path))

    return written


def _extract_with_pdfplumber(path: Path) -> dict[int, str]:
    try:
        import pdfplumber
    except ImportError:
        return {}

    texts: dict[int, str] = {}
    try:
        with pdfplumber.open(str(path)) as pdf:
            for index, page in enumerate(pdf.pages):
                texts[index] = page.extract_text() or ""
    except Exception:
        return {}
    return texts


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _count_page_images(page: object) -> int:
    try:
        return len(page.images)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        resources = page.get("/Resources")  # type: ignore[attr-defined]
        if resources is None:
            return 0
        resources = resources.get_object()
        xobjects = resources.get("/XObject")
        if xobjects is None:
            return 0
        xobjects = xobjects.get_object()
        count = 0
        for xobject in xobjects.values():
            candidate = xobject.get_object()
            if candidate.get("/Subtype") == "/Image":
                count += 1
        return count
    except Exception:
        return 0
