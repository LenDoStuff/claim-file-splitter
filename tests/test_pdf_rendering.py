from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from claim_file_splitter.pdf import render_pdf_pages


def test_render_pdf_pages_creates_image_data_uris(tmp_path: Path) -> None:
    source_pdf = tmp_path / "claim.pdf"
    _write_pdf(source_pdf, page_count=2)

    rendered = render_pdf_pages(
        source_pdf,
        [1, 2],
        tmp_path / "images",
        dpi=100,
        image_format="jpeg",
        jpeg_quality=80,
        keep_paths=True,
    )

    assert sorted(rendered) == [1, 2]
    first = rendered[1]
    assert first.mime_type == "image/jpeg"
    assert first.data_uri.startswith("data:image/jpeg;base64,")
    assert first.width_px > 0
    assert first.height_px > 0
    assert first.byte_size > 0
    assert first.path is not None
    assert first.path.exists()


def _write_pdf(path: Path, page_count: int) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    for page_number in range(1, page_count + 1):
        pdf.drawString(72, 740, f"Rendered page {page_number}")
        pdf.showPage()
    pdf.save()
