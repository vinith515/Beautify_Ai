"""
Merge IQAC front matter with project report body into final PDF.

Keeps IQAC pages: Title, Certificate, Declaration, Acknowledgement, Abstract
Removes: Vision/Mission, POs, Project Outcomes mapping, List of Figures, Contents template
Removes: Generated report cover page (page 1)
Adds: Border + college/project header on every page
"""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter, PageObject
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from generate_project_report import (
    ATTACHED_DIR,
    COLLEGE_NAME,
    DOCS_DIR,
    PAGE_BORDER_MARGIN,
    PROJECT_NAME,
    generate_report_body,
)

IQAC_DOCX = Path(
    r"c:\Users\vinee\OneDrive\Documents\Real-Time Research  Project Document Format - IQAC - Final .docx"
)
IQAC_FULL_PDF = DOCS_DIR / "iqac_full.pdf"
IQAC_TRIMMED_PDF = DOCS_DIR / "iqac_front_matter.pdf"
REPORT_BODY_PDF = DOCS_DIR / "report_body.pdf"
FINAL_PDF = DOCS_DIR / "Claude_Acne_Model_Project_Report.pdf"

# 1-indexed pages to keep from IQAC PDF (after docx2pdf conversion)
IQAC_KEEP_PAGES = [1, 2, 7, 8, 9, 10]


def convert_iqac_docx():
    if not IQAC_FULL_PDF.exists() or IQAC_FULL_PDF.stat().st_mtime < IQAC_DOCX.stat().st_mtime:
        from docx2pdf import convert
        print("[1/5] Converting IQAC docx to PDF...")
        convert(str(IQAC_DOCX), str(IQAC_FULL_PDF))
    else:
        print("[1/5] Using cached IQAC PDF...")


def trim_iqac_pages():
    reader = PdfReader(str(IQAC_FULL_PDF))
    writer = PdfWriter()
    for page_num in IQAC_KEEP_PAGES:
        writer.add_page(reader.pages[page_num - 1])
    with open(IQAC_TRIMMED_PDF, "wb") as f:
        writer.write(f)
    print(f"      IQAC front matter: {len(IQAC_KEEP_PAGES)} pages kept "
          f"(removed Vision/Mission, POs, Outcomes, Contents)")
    return IQAC_TRIMMED_PDF


def make_page_overlay(page_number: int) -> PageObject:
    buf = BytesIO()
    w, h = A4
    m = PAGE_BORDER_MARGIN
    c = canvas.Canvas(buf, pagesize=A4)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1)
    c.rect(m, m, w - 2 * m, h - 2 * m)
    c.setFont("Helvetica-Bold", 8)
    header = f"{COLLEGE_NAME} | {PROJECT_NAME}"
    c.drawString(m + 10, h - m + 8, header)
    c.drawRightString(w - m - 10, h - m + 8, str(page_number))
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def apply_overlay_to_all_pages(input_pdf: Path, output_pdf: Path):
    reader = PdfReader(str(input_pdf))
    writer = PdfWriter()
    total = len(reader.pages)
    for i, page in enumerate(reader.pages):
        overlay = make_page_overlay(i + 1)
        page.merge_page(overlay)
        writer.add_page(page)
    with open(output_pdf, "wb") as f:
        writer.write(f)
    print(f"      Applied border + header to {total} pages")


def merge_pdfs(iqac_pdf: Path, body_pdf: Path, output_pdf: Path):
    writer = PdfWriter()
    for pdf in (iqac_pdf, body_pdf):
        reader = PdfReader(str(pdf))
        for page in reader.pages:
            writer.add_page(page)
    with open(output_pdf, "wb") as f:
        writer.write(f)


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    ATTACHED_DIR.mkdir(exist_ok=True)

    convert_iqac_docx()
    trim_iqac_pages()

    print("[2/5] Generating report body (no cover, attached diagrams)...")
    generate_report_body(
        output_path=REPORT_BODY_PDF,
        skip_cover=True,
        use_attached=True,
    )

    print("[3/5] Merging IQAC front matter + report body...")
    merged_temp = DOCS_DIR / "_merged_temp.pdf"
    merge_pdfs(IQAC_TRIMMED_PDF, REPORT_BODY_PDF, merged_temp)

    print("[4/5] Adding border and header to all pages...")
    staged = DOCS_DIR / "_final_staged.pdf"
    apply_overlay_to_all_pages(merged_temp, staged)
    merged_temp.unlink(missing_ok=True)
    final_path = FINAL_PDF
    try:
        staged.replace(final_path)
    except PermissionError:
        final_path = DOCS_DIR / "Beautify_AI_Project_Report_Final.pdf"
        staged.replace(final_path)
        print(f"      (Original file locked — saved as {final_path.name})")

    print("[5/5] Done!")
    print(f"\n  Final PDF: {final_path}")
    print(f"  Pages: {len(PdfReader(str(final_path)).pages)}")
    print(f"  Open: docs\\OPEN_REPORT.bat or docs\\view_report.html")


if __name__ == "__main__":
    main()
