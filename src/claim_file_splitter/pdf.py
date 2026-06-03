from __future__ import annotations

import base64
import re
from collections.abc import Iterable
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from .models import DOCUMENT_TYPE_PREFIXES, DocumentSegment, PageFeatures, PageImage
from .models import WrittenDocument


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


def render_pdf_pages(
    input_pdf: str | Path,
    page_numbers: Iterable[int],
    output_dir: str | Path,
    *,
    dpi: int = 160,
    image_format: str = "jpeg",
    jpeg_quality: int = 85,
    keep_paths: bool = False,
) -> dict[int, PageImage]:
    if dpi < 72:
        raise ValueError("dpi must be at least 72.")

    normalized_format = image_format.strip().lower()
    if normalized_format in {"jpg", "jpeg"}:
        extension = "jpg"
        pil_format = "JPEG"
        mime_type = "image/jpeg"
    elif normalized_format == "png":
        extension = "png"
        pil_format = "PNG"
        mime_type = "image/png"
    else:
        raise ValueError("image_format must be 'jpeg' or 'png'.")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    source = Path(input_pdf)

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "Rendering PDF pages as images requires pypdfium2. "
            "Install project dependencies with: python -m pip install -e ."
        ) from exc

    rendered: dict[int, PageImage] = {}
    document = pdfium.PdfDocument(str(source))
    try:
        page_count = len(document)
        for page_number in page_numbers:
            if page_number < 1 or page_number > page_count:
                raise ValueError(
                    f"Page {page_number} is outside PDF page range 1-{page_count}."
                )

            page = document[page_number - 1]
            bitmap = page.render(scale=dpi / 72)
            image = bitmap.to_pil()
            if pil_format == "JPEG":
                image = image.convert("RGB")

            image_path = target_dir / f"page_{page_number:06d}.{extension}"
            save_kwargs = (
                {"quality": jpeg_quality, "optimize": True}
                if pil_format == "JPEG"
                else {}
            )
            image.save(image_path, format=pil_format, **save_kwargs)
            data = image_path.read_bytes()
            data_uri = (
                f"data:{mime_type};base64,"
                f"{base64.b64encode(data).decode('ascii')}"
            )
            rendered[page_number] = PageImage(
                page_number=page_number,
                mime_type=mime_type,
                width_px=image.width,
                height_px=image.height,
                byte_size=len(data),
                data_uri=data_uri,
                path=image_path if keep_paths else None,
            )
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()

    return rendered


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
