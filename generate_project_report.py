"""
Generate comprehensive project report PDF for Beautify AI.
Run: python generate_project_report.py
Output: docs/Claude_Acne_Model_Project_Report.pdf
"""

import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).parent.resolve()
DOCS_DIR = PROJECT_ROOT / "docs"
DIAGRAMS_DIR = DOCS_DIR / "diagrams"
ATTACHED_DIR = DOCS_DIR / "attached"
OUTPUT_PDF = DOCS_DIR / "Claude_Acne_Model_Project_Report.pdf"

COLLEGE_NAME = "KESHAV MEMORIAL INSTITUTE OF TECHNOLOGY"
PROJECT_NAME = "Beautify AI"
PAGE_BORDER_MARGIN = 36  # points (~0.5 inch)


def ensure_dirs():
    DOCS_DIR.mkdir(exist_ok=True)
    DIAGRAMS_DIR.mkdir(exist_ok=True)
    ATTACHED_DIR.mkdir(exist_ok=True)


def attached_path(name):
    p = ATTACHED_DIR / name
    return str(p) if p.exists() else None


def draw_page_template(canvas, doc):
    """Border + college/project header on every content page."""
    from reportlab.lib.pagesizes import A4
    w, h = A4
    m = PAGE_BORDER_MARGIN
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#000000"))
    canvas.setLineWidth(1)
    canvas.rect(m, m, w - 2 * m, h - 2 * m)
    canvas.setFont("Helvetica-Bold", 8)
    header = f"{COLLEGE_NAME} | {PROJECT_NAME}"
    canvas.drawString(m + 10, h - m + 8, header)
    canvas.drawRightString(w - m - 10, h - m + 8, str(canvas.getPageNumber()))
    canvas.restoreState()


# ─── Diagram Generation ──────────────────────────────────────────────────────

def save_fig(fig, name):
    path = DIAGRAMS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def draw_architecture_diagram():
    """Deployed web inference architecture (matches app.py + inference/pipeline.py)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Deployed System Architecture (Web Inference)", fontsize=14, fontweight="bold", pad=15)

    boxes = [
        (5, 8.8, "User Browser", "#E3F2FD"),
        (5, 7.2, "Flask Server — app.py\n(static UI + REST API, port 5000)", "#BBDEFB"),
        (5, 5.6, "POST /api/process_image\n(Base64 JPEG in JSON)", "#90CAF9"),
        (5, 4.0, "AcneRemovalPipeline\ninference/pipeline.py", "#64B5F6"),
        (5, 2.4, "Replicate API — CodeFormer\nsczhou/codeformer", "#C8E6C9"),
        (5, 0.8, "Processed Image (Base64 JPEG)", "#FFF9C4"),
    ]
    for x, y, text, color in boxes:
        box = FancyBboxPatch(
            (x - 2.0, y - 0.5), 4.0, 1.0,
            boxstyle="round,pad=0.05", facecolor=color, edgecolor="#333", linewidth=1.2
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, fontweight="bold")

    for y1, y2 in [(8.3, 7.7), (6.7, 6.1), (5.1, 4.5), (3.5, 2.9), (1.9, 1.3)]:
        ax.annotate("", xy=(5, y2), xytext=(5, y1),
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))

    ax.text(7.2, 4.0, "smooth_strength\n→ codeformer_fidelity", fontsize=7, color="#555", style="italic")
    ax.text(7.2, 2.4, "REPLICATE_API_TOKEN\nfrom .env", fontsize=7, color="#555", style="italic")
    return save_fig(fig, "architecture.png")


def draw_use_case_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Use Case Diagram", fontsize=14, fontweight="bold")

    # System boundary
    rect = FancyBboxPatch((3, 0.5), 8.5, 6.5, boxstyle="round,pad=0.1",
                          facecolor="#F5F5F5", edgecolor="#333", linewidth=2, linestyle="--")
    ax.add_patch(rect)
    ax.text(7.25, 6.7, "Beautify AI System", ha="center", fontsize=10, fontweight="bold")

    # Actor
    ax.plot(1.5, 4, "o", markersize=20, color="#1976D2")
    ax.plot([1.5, 1.5], [3.5, 2.5], "k-", lw=2)
    ax.plot([1.0, 2.0], [3.2, 3.2], "k-", lw=2)
    ax.plot([1.5, 1.0], [2.5, 1.8], "k-", lw=2)
    ax.plot([1.5, 2.0], [2.5, 1.8], "k-", lw=2)
    ax.text(1.5, 1.2, "User", ha="center", fontsize=10, fontweight="bold")

    use_cases = [
        (5.5, 5.5, "Upload Face Image"),
        (8.5, 5.5, "Remove Acne\n(CodeFormer API)"),
        (5.5, 3.5, "View Processed Result"),
        (8.5, 3.5, "Generate Acne Masks\n(mask_generator.py)"),
        (7, 1.5, "Train U-Net /\nStarGAN Models"),
    ]
    for x, y, label in use_cases:
        ellipse = mpatches.Ellipse((x, y), 2.2, 0.8, facecolor="#E8F5E9", edgecolor="#2E7D32", lw=1.5)
        ax.add_patch(ellipse)
        ax.text(x, y, label, ha="center", va="center", fontsize=7)

    for x, y, _ in use_cases[:3]:
        ax.annotate("", xy=(x - 1.0, y), xytext=(2.0, 4),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1))

    return save_fig(fig, "use_case.png")


def draw_sequence_diagram():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Sequence Diagram — Image Processing Flow", fontsize=14, fontweight="bold")

    actors = [("User", 1), ("Frontend", 3), ("Flask API", 5.5), ("Pipeline", 8), ("Replicate AI", 10)]
    for name, x in actors:
        ax.text(x, 9.5, name, ha="center", fontsize=9, fontweight="bold")
        ax.plot([x, x], [0.5, 9], "k--", lw=0.8, alpha=0.5)

    messages = [
        (1, 3, 8.8, "1. Upload image"),
        (3, 5.5, 8.2, "2. POST /api/process_image"),
        (5.5, 8, 7.6, "3. pipe.process(img)"),
        (8, 10, 7.0, "4. CodeFormer API call"),
        (10, 8, 6.4, "5. Restored face image"),
        (8, 5.5, 5.8, "6. Return result dict"),
        (5.5, 3, 5.2, "7. JSON + base64 image"),
        (3, 1, 4.6, "8. Display result"),
    ]
    for x1, x2, y, label in messages:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.2))
        ax.text((x1 + x2) / 2, y + 0.15, label, ha="center", fontsize=7)

    return save_fig(fig, "sequence.png")


def draw_state_diagram():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("State Chart Diagram — Processing States", fontsize=14, fontweight="bold")

    states = [
        (1.5, 3, "Idle"),
        (4, 3, "Image\nReceived"),
        (6.5, 3, "Processing"),
        (9, 3, "AI\nRestoration"),
        (11, 3, "Complete"),
        (6.5, 1, "Error"),
    ]
    for x, y, label in states:
        if label == "Error":
            circle = mpatches.FancyBboxPatch((x - 0.7, y - 0.4), 1.4, 0.8,
                                             boxstyle="round", facecolor="#FFCDD2", edgecolor="#C62828", lw=2)
        else:
            circle = mpatches.Circle((x, y), 0.55, facecolor="#E3F2FD", edgecolor="#1565C0", lw=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold")

    transitions = [
        (2.05, 3, 3.45, 3, "upload"),
        (4.55, 3, 5.95, 3, "validate"),
        (7.05, 3, 8.45, 3, "call API"),
        (9.55, 3, 10.45, 3, "success"),
        (6.5, 2.55, 6.5, 1.45, "failure"),
        (6.5, 0.55, 1.5, 2.55, "retry"),
    ]
    for x1, y1, x2, y2, label in transitions:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.2, label, ha="center", fontsize=7)

    return save_fig(fig, "state_chart.png")


def draw_deployment_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Deployment Diagram", fontsize=14, fontweight="bold")

    nodes = [
        (2, 6, "Client Device\n(Browser)", "#E3F2FD", 2.5, 1.2),
        (5, 6, "Application Server\n(Flask :5000)", "#C8E6C9", 2.8, 1.2),
        (8, 6, "Replicate Cloud\n(CodeFormer)", "#FFF9C4", 2.5, 1.2),
        (2, 3, "Local GPU\n(RTX 3050)", "#FFE0B2", 2.5, 1.2),
        (5, 3, "Google Colab\n(StarGAN Training)", "#F3E5F5", 2.8, 1.2),
        (8, 3, "Checkpoints\nStorage", "#ECEFF1", 2.5, 1.2),
    ]
    for x, y, label, color, w, h in nodes:
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor="#333", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold")

    connections = [(2, 5.4, 5, 5.4), (5, 5.4, 8, 5.4), (5, 5.4, 5, 3.6), (2, 3.6, 5, 3.6), (8, 5.4, 8, 3.6)]
    for x1, y1, x2, y2 in connections:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))

    ax.text(3.5, 5.7, "HTTP", fontsize=7, color="#555")
    ax.text(6.5, 5.7, "HTTPS/API", fontsize=7, color="#555")
    return save_fig(fig, "deployment.png")


def draw_pipeline_flow():
    """Training pipeline documented in Readme.md — implemented in this project."""
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_title("ML Training Pipeline (Local + Colab)", fontsize=13, fontweight="bold")

    steps = [
        ("dataset/Acne/ — 419 acne face images", "#E3F2FD"),
        ("utils/mask_generator.py\nHSV thresholding → dataset/masks/", "#E3F2FD"),
        ("training/train_unet.py\n50 epochs, RTX 3050 → checkpoints/unet.pth", "#FFE0B2"),
        ("U-Net predicts binary acne mask (256×256)", "#FFE0B2"),
        ("training/train_stargan.py on Google Colab\n→ checkpoints/generator (1).pth", "#F3E5F5"),
        ("StarGAN: acne domain → clear skin (CelebA-HQ)", "#F3E5F5"),
        ("Optional: identity_preservation.py, gfpgan_enhancer.py", "#FFF9C4"),
        ("Web inference uses CodeFormer via Replicate (inference/pipeline.py)", "#C8E6C9"),
    ]
    y = 12
    for step, color in steps:
        box = FancyBboxPatch((0.8, y - 0.55), 6.4, 1.0, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor="#1565C0", linewidth=1.2)
        ax.add_patch(box)
        ax.text(4, y, step, ha="center", va="center", fontsize=7.5, fontweight="bold")
        if y > 1.5:
            ax.annotate("", xy=(4, y - 0.7), xytext=(4, y - 0.55),
                        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2))
        y -= 1.45

    return save_fig(fig, "pipeline_flow.png")


def collect_project_images():
    """Gather real project images only — no synthetic test outputs."""
    images = []

    acne_dir = PROJECT_ROOT / "dataset" / "Acne" / "Acne"
    if acne_dir.exists():
        samples = sorted(acne_dir.glob("*.jpg"))[:2]
        for p in samples:
            images.append((f"Dataset Acne Image — {p.name}", str(p)))

    mask_dir = PROJECT_ROOT / "dataset" / "masks"
    if mask_dir.exists():
        mask = next(iter(sorted(mask_dir.glob("*.png"))), None)
        if mask:
            images.append((f"Generated Acne Mask — {mask.name}", str(mask)))

    unet_vis = PROJECT_ROOT / "outputs" / "unet_vis" / "epoch_050.png"
    if unet_vis.exists():
        images.append(("U-Net Training Result — Epoch 50 (outputs/unet_vis/)", str(unet_vis)))

    stargan_vis = PROJECT_ROOT / "outputs" / "stargan_vis" / "epoch_001.png"
    if stargan_vis.exists():
        images.append(("StarGAN Training Result — Epoch 1 (outputs/stargan_vis/)", str(stargan_vis)))

    celeba = PROJECT_ROOT / "dataset" / "celeba_hq" / "train"
    if celeba.exists():
        clean = next(iter(sorted(celeba.glob("*.jpg"))), None) or next(iter(sorted(celeba.glob("*.png"))), None)
        if clean:
            images.append((f"Clean Reference Image (CelebA-HQ) — {clean.name}", str(clean)))

    return images


# ─── PDF Builder ─────────────────────────────────────────────────────────────

class ReportBuilder:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        self.story = []
        self.toc_entries = []

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            name="CoverTitle", fontSize=26, leading=32, alignment=TA_CENTER,
            fontName="Helvetica-Bold", spaceAfter=20, textColor=colors.HexColor("#1565C0"),
        ))
        self.styles.add(ParagraphStyle(
            name="CoverSub", fontSize=14, leading=18, alignment=TA_CENTER,
            fontName="Helvetica", spaceAfter=8,
        ))
        self.styles.add(ParagraphStyle(
            name="Chapter", fontSize=18, leading=22, alignment=TA_LEFT,
            fontName="Helvetica-Bold", spaceBefore=20, spaceAfter=12,
            textColor=colors.HexColor("#0D47A1"),
        ))
        self.styles.add(ParagraphStyle(
            name="Section", fontSize=14, leading=18, alignment=TA_LEFT,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=8,
            textColor=colors.HexColor("#1565C0"),
        ))
        self.styles.add(ParagraphStyle(
            name="SubSection", fontSize=12, leading=15, alignment=TA_LEFT,
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name="Body", fontSize=11, leading=15, alignment=TA_JUSTIFY,
            fontName="Helvetica", spaceAfter=8,
        ))
        self.styles.add(ParagraphStyle(
            name="TOCChapter", fontSize=12, leading=16, fontName="Helvetica-Bold", spaceAfter=4,
        ))
        self.styles.add(ParagraphStyle(
            name="TOCItem", fontSize=11, leading=14, fontName="Helvetica",
            leftIndent=20, spaceAfter=2,
        ))

    def add(self, *elements):
        for el in elements:
            if isinstance(el, list):
                self.story.extend(el)
            else:
                self.story.append(el)

    def p(self, text, style="Body"):
        self.add(Paragraph(text, self.styles[style]))

    def heading(self, text, level=1):
        style = {1: "Chapter", 2: "Section", 3: "SubSection"}[level]
        self.add(Paragraph(text, self.styles[style]))

    def spacer(self, h=0.3):
        self.add(Spacer(1, h * cm))

    def page_break(self):
        self.add(PageBreak())

    def image(self, path, width=14 * cm, height=None):
        if path and os.path.exists(path):
            if height is None:
                from PIL import Image as PILImage
                with PILImage.open(path) as pil:
                    pw, ph = pil.size
                height = width * (ph / pw)
            img = Image(path, width=width, height=height)
            img.hAlign = "CENTER"
            self.add(Spacer(1, 0.2 * cm), img, Spacer(1, 0.3 * cm))
        else:
            self.p(f"<i>[Image not found: {path}]</i>")

    def table(self, data, col_widths=None):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        self.add(t, Spacer(1, 0.3 * cm))

    def build_cover(self):
        self.add(
            Spacer(1, 4 * cm),
            Paragraph("CLAUDE ACNE MODEL", self.styles["CoverTitle"]),
            Paragraph("AI-Based Acne Removal Pipeline", self.styles["CoverSub"]),
            Paragraph("U-Net + StarGAN Training | CodeFormer Web Inference", self.styles["CoverSub"]),
            Spacer(1, 2 * cm),
            Paragraph("Project Report", self.styles["CoverSub"]),
            Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", self.styles["CoverSub"]),
            Spacer(1, 3 * cm),
            Paragraph(
                "Stack: PyTorch · OpenCV · Flask · Replicate CodeFormer · BeautifyAI UI",
                self.styles["CoverSub"],
            ),
            PageBreak(),
        )

    def build_toc(self):
        self.heading("CONTENTS", 1)
        self.spacer(0.5)
        toc = [
            ("CHAPTER-1", "1. INTRODUCTION", [
                "1.1 Purpose of the Project", "1.2 Problem with Existing Systems",
                "1.3 Proposed System", "1.4 Scope of the Project", "1.5 Architecture Diagram",
            ]),
            ("CHAPTER-2", "2. LITERATURE SURVEY", []),
            ("CHAPTER-3", "3. SOFTWARE REQUIREMENT SPECIFICATION", [
                "3.1 Introduction to SRS", "3.2 Role of SRS",
                "3.3 Requirements Specification Document", "3.4 Functional Requirements",
                "3.5 Non-Functional Requirements", "3.6 Performance Requirements",
                "3.7 Software Requirements", "3.8 Hardware Requirements",
            ]),
            ("CHAPTER-4", "4. SYSTEM DESIGN", [
                "4.1 Introduction to UML", "4.2 UML Diagrams",
                "4.2.1 Use Case Diagram", "4.2.2 Sequence Diagram",
                "4.2.3 State Chart Diagram", "4.2.4 Deployment Diagram",
                "4.3 Technologies Used",
            ]),
            ("CHAPTER-5", "5. IMPLEMENTATION", [
                "5.1 Setting Up the Development Environment",
                "5.2 Coding the Logic", "5.3 Connecting the Dashboard",
                "5.4 Screenshots", "5.5 UI Screenshots",
            ]),
            ("CHAPTER-6", "6. SOFTWARE TESTING", [
                "6.1 Introduction", "6.1.1 Testing Objectives",
                "6.1.2 Testing Strategies", "6.1.3 System Evaluation",
                "6.1.4 Testing New System", "6.2 Test Cases",
            ]),
            ("", "CONCLUSION", []),
            ("", "FUTURE ENHANCEMENTS", []),
            ("", "REFERENCES", []),
            ("", "BIBLIOGRAPHY", []),
        ]
        for chapter, title, items in toc:
            self.p(f"<b>{chapter}</b>  {title}", "TOCChapter")
            for item in items:
                self.p(item, "TOCItem")
        self.page_break()

    def build_chapter1(self, arch_path):
        self.heading("CHAPTER-1", 1)
        self.heading("1. INTRODUCTION", 2)

        self.heading("1.1 Purpose of the Project", 3)
        self.p(
            "The Beautify AI is a deep-learning project that builds and trains custom models "
            "for acne detection and skin restoration, then exposes acne removal through a Flask web API. "
            "The project has two goals: (1) train a U-Net segmentation model and a StarGAN image-translation "
            "model on a local acne dataset (419 images) paired with CelebA-HQ clean faces; and "
            "(2) provide a working web service where users upload a face photo and receive an "
            "AI-restored result via the CodeFormer model on Replicate."
        )
        self.p(
            "This report documents the actual implementation in this repository — file structure, "
            "trained checkpoints, training scripts, API endpoints, and test suite — not a generic template."
        )

        self.heading("1.2 Problem with Existing Systems", 3)
        self.p(
            "Traditional photo editing tools such as Adobe Photoshop or mobile filter applications "
            "require manual selection and retouching of each blemish, which is time-consuming and "
            "demands professional skill. Generic beauty filters often over-smooth the entire face, "
            "removing natural skin pores and altering facial features, resulting in an unnatural "
            "\"plastic\" appearance."
        )
        self.p(
            "Existing automated solutions face several limitations: (1) Rule-based HSV thresholding "
            "alone produces noisy masks and fails on varied skin tones; (2) Simple inpainting methods "
            "cannot handle severe cystic acne; (3) GAN-based approaches without identity preservation "
            "may change the person's facial structure; (4) Cloud-only solutions lack offline training "
            "capability; and (5) Most commercial apps do not expose their underlying ML pipeline "
            "for academic or customization purposes."
        )

        self.heading("1.3 Proposed System", 3)
        self.p(
            "<b>Training subsystem (implemented locally):</b> Acne images in dataset/Acne/Acne/ are "
            "processed by utils/mask_generator.py using HSV colour thresholding to produce binary masks "
            "in dataset/masks/ (413 masks generated). A U-Net model (models/unet.py) is trained via "
            "training/train_unet.py for 50 epochs on an RTX 3050 GPU, saving checkpoints/unet.pth. "
            "A StarGAN generator (models/stargan.py) is trained on Google Colab using acne and "
            "CelebA-HQ images, saving checkpoints/generator (1).pth."
        )
        self.p(
            "<b>Web inference subsystem (currently deployed):</b> app.py runs a Flask server on port 5000. "
            "The /api/process_image endpoint receives a base64-encoded image, passes it to "
            "AcneRemovalPipeline in inference/pipeline.py, which calls the CodeFormer model on Replicate "
            "(model ID: sczhou/codeformer). The restored image is returned as base64 JPEG. "
            "Optional modules — models/identity_preservation.py and models/gfpgan_enhancer.py — are "
            "implemented for the full local pipeline described in Readme.md but are not wired into "
            "the current CodeFormer-based inference path."
        )

        self.heading("1.4 Scope of the Project", 3)
        self.p("The project scope encompasses the following components:")
        scopes = [
            ["Component", "Details in This Project", "Status"],
            ["Acne Dataset", "419 images in dataset/Acne/Acne/", "Present"],
            ["Mask Dataset", "413 HSV-generated masks in dataset/masks/", "Present"],
            ["CelebA-HQ", "Clean face images in dataset/celeba_hq/", "Present"],
            ["U-Net Training", "50 epochs completed; checkpoints/unet.pth saved", "Trained"],
            ["StarGAN Training", "Generator saved as checkpoints/generator (1).pth", "Trained"],
            ["Web API", "Flask app.py — POST /api/process_image", "Implemented"],
            ["CodeFormer Inference", "inference/pipeline.py via Replicate API", "Implemented"],
            ["Test Suite", "test_pipeline.py — 8 component tests", "Implemented"],
            ["BeautifyAI UI", "Configured in app.py static_folder (requires built dist/)", "Configured"],
        ]
        self.table(scopes, [3.5 * cm, 7 * cm, 2.5 * cm])

        self.p("Out of scope for this project: medical diagnosis, dermatology advice, real-time video processing.")

        self.heading("1.5 Architecture Diagram", 3)
        self.p(
            "The diagram below shows the <b>deployed web inference path</b> actually used when "
            "running python app.py. The ML training pipeline (U-Net, StarGAN, mask generation) "
            "runs separately via the training/ and utils/ modules."
        )
        self.image(arch_path, 15 * cm)
        self.page_break()

    def build_chapter2(self, pipeline_path):
        self.heading("CHAPTER-2", 1)
        self.heading("2. LITERATURE SURVEY", 2)

        literature = [
            ("U-Net (Ronneberger et al., 2015)", "Introduced encoder-decoder architecture with skip connections for biomedical image segmentation. Widely adopted for skin lesion and acne region detection due to precise pixel-level localization."),
            ("StarGAN (Choi et al., 2018)", "Proposed a unified generative adversarial network for multi-domain image-to-image translation using a single generator. Adapted in this project for acne-to-clear-skin domain transfer."),
            ("GFPGAN (Wang et al., 2021)", "Generative facial prior GAN for blind face restoration. Used as the final enhancement stage to recover fine facial details after acne removal."),
            ("CodeFormer (Zhou et al., 2022)", "Transformer-based face restoration model using codebook lookup for robust identity-preserving restoration. Deployed via Replicate API for production inference."),
            ("FaceNet (Schroff et al., 2015)", "Deep learning system that maps faces to Euclidean space for identity verification. Cosine similarity threshold applied to ensure output identity matches input."),
            ("RetinaFace (Deng et al., 2020)", "Single-stage face detector with multi-task learning. Used for face detection and alignment before segmentation."),
            ("Pix2Pix (Isola et al., 2017)", "Conditional GAN for image-to-image translation. Evaluated but StarGAN chosen for multi-domain flexibility."),
            ("HSV Color Space Analysis", "Traditional computer vision approach for detecting reddish acne regions. Used as baseline mask generation before U-Net training."),
        ]
        for title, desc in literature:
            self.p(f"<b>{title}:</b> {desc}")

        self.spacer(0.5)
        self.p(
            "This project applies U-Net and StarGAN for custom model training on the project's own acne "
            "dataset, and uses CodeFormer (Zhou et al., 2022) for the live web API because it provides "
            "reliable face restoration without requiring local GPU inference at request time."
        )
        self.image(pipeline_path, 12 * cm)
        self.page_break()

    def build_chapter3(self):
        self.heading("CHAPTER-3", 1)
        self.heading("3. SOFTWARE REQUIREMENT SPECIFICATION", 2)

        self.heading("3.1 Introduction to SRS", 3)
        self.p(
            "This Software Requirements Specification (SRS) document defines the functional and "
            "non-functional requirements for the Beautify AI system. It serves as a contract "
            "between developers and stakeholders, ensuring all features are clearly specified before "
            "implementation."
        )

        self.heading("3.2 Role of SRS", 3)
        self.p(
            "The SRS establishes a baseline for development, testing, and validation. It enables "
            "traceability from requirements to test cases, facilitates communication among team members, "
            "and provides criteria for acceptance testing of the delivered system."
        )

        self.heading("3.3 Requirements Specification Document", 3)
        self.p(
            "The system is specified as a web-based acne removal application with optional local "
            "ML training capabilities. Users interact through a browser interface; the backend "
            "processes images via REST API calls to the AI pipeline."
        )

        self.heading("3.4 Functional Requirements", 3)
        fr = [
            ["ID", "Requirement (from actual code)", "Module"],
            ["FR-01", "Accept base64 image via POST /api/process_image", "app.py"],
            ["FR-02", "Decode image to BGR numpy array with OpenCV", "app.py"],
            ["FR-03", "Call CodeFormer on Replicate with fidelity parameter", "inference/pipeline.py"],
            ["FR-04", "Return processed image as base64 JPEG in JSON response", "app.py"],
            ["FR-05", "Return timing dict and identity_similarity in response", "app.py"],
            ["FR-06", "Fall back to original image if CodeFormer API fails", "inference/pipeline.py"],
            ["FR-07", "Generate binary acne masks using HSV thresholding", "utils/mask_generator.py"],
            ["FR-08", "Train U-Net with BCE+Dice loss, IoU validation, fp16", "training/train_unet.py"],
            ["FR-09", "Train StarGAN G/D with adversarial + domain losses", "training/train_stargan.py"],
            ["FR-10", "Load paired acne/mask data with augmentation", "utils/dataset.py"],
            ["FR-11", "Serve static frontend from BeautifyAI dist folder", "app.py"],
            ["FR-12", "Validate all 8 ML components with synthetic data", "test_pipeline.py"],
        ]
        self.table(fr, [1.2 * cm, 8.8 * cm, 3 * cm])

        self.heading("3.5 Non-Functional Requirements", 3)
        nfr = [
            ["ID", "Requirement", "Metric"],
            ["NFR-01", "Usability: Intuitive UI requiring no technical knowledge", "≤ 3 clicks to result"],
            ["NFR-02", "Reliability: Graceful error handling on invalid images", "100% error responses"],
            ["NFR-03", "Scalability: API supports concurrent requests", "Flask threaded mode"],
            ["NFR-04", "Maintainability: Modular code structure", "Separate models/utils/inference"],
            ["NFR-05", "Security: API token stored in environment variables", "No hardcoded secrets"],
            ["NFR-06", "Portability: Cross-platform Python 3.10+", "Windows/Linux/macOS"],
        ]
        self.table(nfr, [1.5 * cm, 8 * cm, 3.5 * cm])

        self.heading("3.6 Performance Requirements", 3)
        self.p(
            "From app.py: MAX_CONTENT_LENGTH = 50 MB per upload. CodeFormer processing time is logged "
            "in result['timing']['ai_restore'] and returned to the client. U-Net training uses "
            "batch_size=4 with fp16 mixed precision on a 4 GB GPU (documented in train_unet.py). "
            "Replicate API latency depends on network and cloud queue — typically several seconds per image."
        )

        self.heading("3.7 Software Requirements", 3)
        sw = [
            ["Category", "Technology", "Source"],
            ["Core", "Python 3, PyTorch ≥2.0, torchvision ≥0.15", "requirements.txt"],
            ["Vision", "opencv-python ≥4.8, Pillow ≥10, numpy ≥1.24", "requirements.txt"],
            ["Plots", "matplotlib ≥3.7", "requirements.txt"],
            ["Web API", "Flask, Flask-CORS, python-dotenv", "app.py"],
            ["Cloud AI", "replicate, requests", "inference/pipeline.py"],
            ["Optional", "gfpgan, facenet-pytorch, retina-face", "Readme.md"],
            ["Frontend", "BeautifyAI React build (dist/)", "app.py static_folder"],
        ]
        self.table(sw, [3 * cm, 5.5 * cm, 4.5 * cm])

        self.heading("3.8 Hardware Requirements", 3)
        hw = [
            ["Component", "Minimum", "Recommended"],
            ["CPU", "Intel i5 / AMD Ryzen 5", "Intel i7 / AMD Ryzen 7"],
            ["RAM", "8 GB", "16 GB"],
            ["GPU (Training)", "NVIDIA GTX 1650 (4GB VRAM)", "NVIDIA RTX 3050 (4GB VRAM)"],
            ["GPU (Inference)", "Not required (cloud API)", "Optional for local models"],
            ["Storage", "10 GB free space", "50 GB (with datasets)"],
            ["Network", "Broadband internet", "Required for Replicate API"],
            ["Cloud (StarGAN)", "Google Colab T4 GPU", "Google Colab Pro"],
        ]
        self.table(hw, [3.5 * cm, 5 * cm, 4.5 * cm])
        self.page_break()

    def build_chapter4(self, diagrams):
        self.heading("CHAPTER-4", 1)
        self.heading("4. SYSTEM DESIGN", 2)

        self.heading("4.1 Introduction to UML", 3)
        self.p(
            "Unified Modeling Language (UML) is used to visually represent the Beautify AI system "
            "architecture, data flow, database schema, and ML processing pipeline."
        )

        self.heading("4.2 UML Diagrams", 2)

        self.heading("4.2.1 System Architecture Diagram", 3)
        self.p(
            "Three-tier architecture: React Client Browser, Express.js Server Backend "
            "(router, SQLite DB), and FastAPI ML Engine (RetinaFace, BlemishNet, StarGAN, "
            "LaMa Inpainting, DermAnalyzer)."
        )
        self.image(diagrams.get("architecture"), 14 * cm)

        self.heading("4.2.2 Sequence Diagram", 3)
        self.p(
            "User uploads image → React POST /api/beautify/process → Node.js forwards to "
            "FastAPI /process → 3-stage AI pipeline (RetinaFace & BlemishNet, StarGAN & LaMa, "
            "LAB Color Recovery) → result saved to SQLite → comparison slider rendered."
        )
        self.image(diagrams.get("sequence"), 14 * cm)

        self.heading("4.2.3 Database Schema", 3)
        self.p(
            "SQLite database with users table (id, username, email, password, created_at) and "
            "history table (id, user_id FK, original_filename, original_path, result_path, created_at). "
            "One user has many upload history records with ON DELETE CASCADE."
        )
        self.image(diagrams.get("database_schema"), 12 * cm)

        self.heading("4.2.4 ML Processing Pipeline", 3)
        self.p(
            "End-to-end pipeline: Input Image → Face detection & alignment (RetinaFace) → "
            "Acne segmentation (U-Net) → Image translation (StarGAN) → Identity preservation "
            "(ArcFace) → Face enhancement (GFPGAN) → Final acne-free output."
        )
        self.image(diagrams.get("pipeline"), 10 * cm)

        self.heading("4.3 Technologies Used", 2)
        tech = [
            ["File / Module", "Technology", "Role in This Project"],
            ["app.py", "Flask + CORS", "Web server, /api/process_image endpoint"],
            ["inference/pipeline.py", "Replicate CodeFormer", "Live acne removal inference"],
            ["models/unet.py", "PyTorch U-Net", "Acne segmentation (trained, 50 epochs)"],
            ["models/stargan.py", "PyTorch StarGAN v1", "Acne→clear skin GAN (trained)"],
            ["models/gfpgan_enhancer.py", "GFPGAN wrapper", "Optional face enhancement module"],
            ["models/identity_preservation.py", "FaceNet / LightEmbeddingNet", "Optional identity check"],
            ["utils/mask_generator.py", "OpenCV HSV", "Training mask generation"],
            ["utils/dataset.py", "PyTorch DataLoader", "U-Net and StarGAN data loading"],
            ["training/train_unet.py", "PyTorch AMP fp16", "Local U-Net training loop"],
            ["training/train_stargan.py", "PyTorch GAN losses", "Colab StarGAN training loop"],
            ["test_pipeline.py", "Synthetic unit tests", "8-component validation suite"],
        ]
        self.table(tech, [3.5 * cm, 3.5 * cm, 6 * cm])
        self.page_break()

    def build_chapter5(self, screenshots, diagrams=None):
        diagrams = diagrams or {}
        self.heading("CHAPTER-5", 1)
        self.heading("5. IMPLEMENTATION", 2)

        self.heading("5.1 Setting Up the Development Environment", 3)
        if diagrams.get("project_structure"):
            self.p("Project directory structure:")
            self.image(diagrams.get("project_structure"), 14 * cm)
        self.p("Project root structure (from Readme.md and repository):")
        structure = [
            ["Path", "Contents"],
            ["dataset/Acne/Acne/", "419 acne face images (.jpg)"],
            ["dataset/masks/", "413 auto-generated binary masks (.png)"],
            ["dataset/celeba_hq/", "Clean CelebA-HQ train/val images"],
            ["models/", "unet.py, stargan.py, identity_preservation.py, gfpgan_enhancer.py"],
            ["training/", "train_unet.py, train_stargan.py"],
            ["utils/", "mask_generator.py, dataset.py"],
            ["inference/", "pipeline.py (CodeFormer API wrapper)"],
            ["checkpoints/", "unet.pth, generator (1).pth, unet_latest.pth"],
            ["outputs/unet_vis/", "U-Net epoch visualisations (epoch_001 to epoch_050)"],
            ["outputs/stargan_vis/", "StarGAN training visualisations"],
            ["app.py", "Flask web server entry point"],
            ["test_pipeline.py", "Component validation tests"],
        ]
        self.table(structure, [4.5 * cm, 10.5 * cm])

        self.p("Setup steps used in this project:")
        steps = [
            "pip install -r requirements.txt",
            "pip install flask flask-cors python-dotenv replicate requests  (web inference)",
            "Set REPLICATE_API_TOKEN in .env file",
            "python utils/mask_generator.py --input dataset/Acne/Acne/ --output dataset/masks/",
            "python training/train_unet.py --acne_dir dataset/Acne/Acne/ --mask_dir dataset/masks/ --epochs 50 --batch_size 4",
            "python training/train_stargan.py on Google Colab with dataset/celeba_hq/",
            "python test_pipeline.py  (validate all components)",
            "python app.py  → server at http://0.0.0.0:5000",
        ]
        for step in steps:
            self.p(f"• {step}")

        self.heading("5.2 Coding the Logic", 3)
        self.p(
            "<b>Web inference (inference/pipeline.py):</b> AcneRemovalPipeline.__init__ reads "
            "REPLICATE_API_TOKEN from the environment and sets fidelity = 1.0 - smooth_strength "
            "(app.py uses smooth_strength=0.8). process() encodes the image as a JPEG data URI, "
            "calls replicate.run('sczhou/codeformer:7de2ea26...') with codeformer_fidelity, "
            "face_upsample=True, and returns {'output', 'timing', 'identity_similarity', 'mask'}."
        )
        self.p(
            "<b>Flask API (app.py):</b> On startup, loads AcneRemovalPipeline(smooth_strength=0.8). "
            "POST /api/process_image decodes base64 → cv2.imdecode → pipe.process() → "
            "cv2.imencode JPEG quality 95 → JSON response with success, image, timing, "
            "identity_similarity, spots_detected."
        )
        self.p(
            "<b>U-Net training (training/train_unet.py):</b> Uses CombinedLoss (BCE + Dice), "
            "Adam optimizer, ReduceLROnPlateau scheduler, GradScaler fp16. Saves best model to "
            "checkpoints/unet.pth and visualisations to outputs/unet_vis/ each epoch."
        )
        self.p(
            "<b>StarGAN (models/stargan.py):</b> Generator takes image + 2-domain one-hot label; "
            "Discriminator is PatchGAN with domain classifier. Domain 0 = Acne, Domain 1 = Clear skin."
        )
        models = [
            ["Model", "File", "I/O Shape", "Checkpoint"],
            ["U-Net", "models/unet.py", "(B,3,256,256)→(B,1,256,256)", "checkpoints/unet.pth"],
            ["StarGAN-G", "models/stargan.py", "(B,3,256,256)→(B,3,256,256)", "generator (1).pth"],
            ["StarGAN-D", "models/stargan.py", "(B,3,256,256)→patches+domain", "checkpoint (1).pth"],
            ["CodeFormer", "inference/pipeline.py", "BGR image → restored BGR", "Replicate cloud"],
        ]
        self.table(models, [2.2 * cm, 3.8 * cm, 4.5 * cm, 3.5 * cm])

        self.heading("5.3 Connecting the Dashboard", 3)
        self.p(
            "app.py serves the BeautifyAI React frontend from "
            "beautifyai modified/beautifyai/beautifyai/dist as the Flask static_folder. "
            "CORS is enabled. Non-API routes return index.html for SPA routing. "
            "The API contract used by the frontend:"
        )
        api = [
            ["Endpoint", "Method", "Request", "Response"],
            ["/api/process_image", "POST", '{"image": "data:image/jpeg;base64,..."}', '{"success": true, "image": "...", "timing": {...}}'],
            ["/", "GET", "—", "Serves index.html (BeautifyAI UI)"],
        ]
        self.table(api, [3.5 * cm, 1.5 * cm, 5 * cm, 4 * cm])

        self.heading("5.4 Screenshots", 3)
        self.p(
            "Images below are taken directly from this project's dataset and training outputs — "
            "not synthetic demo files."
        )
        for label, path in screenshots:
            self.p(f"<b>{label}</b>")
            h = 8 * cm if "unet_vis" in path or "stargan_vis" in path else None
            self.image(path, 12 * cm, height=h)

        self.heading("5.5 UI Screenshots", 3)
        self.p(
            "The BeautifyAI React UI is integrated via app.py static file serving. "
            "To capture UI screenshots: build the BeautifyAI dist folder, run python app.py, "
            "open http://localhost:5000, upload a face image from dataset/Acne/Acne/, and "
            "the UI will call /api/process_image. test_api.py demonstrates the API call programmatically."
        )
        self.page_break()

    def build_chapter6(self):
        self.heading("CHAPTER-6", 1)
        self.heading("6. SOFTWARE TESTING", 2)

        self.heading("6.1 Introduction", 2)
        self.heading("6.1.1 Testing Objectives", 3)
        self.p(
            "The primary testing objectives are: (1) Verify all pipeline components function correctly "
            "in isolation; (2) Validate end-to-end integration from image input to processed output; "
            "(3) Ensure API endpoints return correct responses for valid and invalid inputs; "
            "(4) Confirm model architectures produce correct tensor shapes; (5) Measure processing "
            "time and identity preservation metrics."
        )

        self.heading("6.1.2 Testing Strategies", 3)
        self.p(
            "A multi-level testing strategy is employed: Unit Testing (individual model forward passes), "
            "Component Testing (mask generator, dataset loaders, loss functions), Integration Testing "
            "(full pipeline with mock weights), API Testing (HTTP endpoint validation), and "
            "System Testing (end-to-end with real images via Replicate API)."
        )

        self.heading("6.1.3 System Evaluation", 3)
        self.p(
            "U-Net training evaluation uses IoU (compute_iou in train_unet.py) on the validation set. "
            "Training visualisations in outputs/unet_vis/ show predicted masks vs. ground truth "
            "across 50 epochs. StarGAN evaluation uses adversarial, domain classification, and "
            "reconstruction losses (models/stargan.py). Web API evaluation uses test_api.py "
            "which POSTs a 100×100 test image and checks HTTP status and JSON response."
        )

        self.heading("6.1.4 Testing New System", 3)
        self.p(
            "Run python test_pipeline.py before training. It executes 8 tests with synthetic data "
            "on CPU or CUDA: (1) Mask Generator, (2) U-Net Model, (3) Dataset Loaders, "
            "(4) StarGAN, (5) Identity Preservation, (6) GFPGAN Enhancer, (7) Training Losses, "
            "(8) Pipeline Integration. For the live API, start app.py and run test_api.py."
        )

        self.heading("6.2 Test Cases", 2)
        tc = [
            ["#", "Test (test_pipeline.py)", "Validates", "Script Function"],
            ["1", "Mask Generator", "256×256 binary mask from HSV", "test_mask_generator()"],
            ["2", "U-Net Model", "Forward pass, predict_mask shapes", "test_unet()"],
            ["3", "Dataset Loaders", "AcneSegmentationDataset, DataLoader", "test_datasets()"],
            ["4", "StarGAN", "G/D forward, all loss functions", "test_stargan()"],
            ["5", "Identity Preservation", "Embeddings, cosine similarity", "test_identity()"],
            ["6", "GFPGAN Enhancer", "enhance_batch output shape", "test_gfpgan()"],
            ["7", "Training Losses", "DiceLoss, CombinedLoss, IoU", "test_training_losses()"],
            ["8", "Pipeline Integration", "FaceDetector, process() keys", "test_pipeline_integration()"],
            ["9", "API Endpoint", "POST /api/process_image returns 200", "test_api.py"],
            ["10", "Beautify Test", "Synthetic acne image processing", "test_beautify.py"],
        ]
        self.table(tc, [0.8 * cm, 3.5 * cm, 5 * cm, 4.7 * cm])
        self.page_break()

    def build_conclusion(self):
        self.heading("CONCLUSION", 1)
        self.p(
            "The Beautify AI project delivers a complete acne-removal ML workflow: 419 acne images "
            "and 413 HSV-generated masks were used to train a U-Net segmentation model for 50 epochs "
            "(checkpoints/unet.pth), and a StarGAN generator was trained for acne-to-clear-skin translation "
            "(checkpoints/generator (1).pth). Training visualisations are saved in outputs/unet_vis/ "
            "and outputs/stargan_vis/."
        )
        self.p(
            "For live use, inference/pipeline.py integrates Replicate's CodeFormer model, and app.py "
            "exposes it as a Flask REST API at /api/process_image. The BeautifyAI frontend is wired "
            "through Flask static file serving. The test_pipeline.py suite validates all eight ML "
            "components. This project demonstrates both custom model training on a real acne dataset "
            "and cloud-based AI deployment through a web interface."
        )
        self.page_break()

    def build_future(self):
        self.heading("FUTURE ENHANCEMENTS", 1)
        enhancements = [
            "Wire trained U-Net + StarGAN checkpoints into inference/pipeline.py for fully local inference (no Replicate dependency).",
            "Activate identity_preservation.py and gfpgan_enhancer.py in the live pipeline as documented in Readme.md.",
            "Complete BeautifyAI frontend build and add before/after comparison UI screenshots.",
            "Extend StarGAN training beyond epoch 1 (outputs/stargan_vis/ currently has epoch_001 only).",
            "Add batch inference endpoint for processing entire dataset/Acne/Acne/ folder.",
            "Deploy Flask app with gunicorn/nginx for production instead of development server.",
            "Add IoU and perceptual quality metrics dashboard for comparing U-Net vs. CodeFormer results.",
            "Expand acne dataset beyond 419 images for improved segmentation generalisation.",
        ]
        for i, item in enumerate(enhancements, 1):
            self.p(f"{i}. {item}")
        self.page_break()

    def build_references(self):
        self.heading("REFERENCES", 1)
        refs = [
            "[1] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.",
            "[2] Choi, Y., Choi, M., Kim, M., Ha, J. W., Kim, S., & Choo, J. (2018). StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation. CVPR 2018.",
            "[3] Wang, X., Li, Y., Zhang, W., & Shan, Y. (2021). Towards Real-World Blind Face Restoration with Generative Facial Prior. CVPR 2021.",
            "[4] Zhou, S., Chen, C., Wang, X., & Loy, C. C. (2022). CodeFormer: Robust Face Restoration and Enhancement with Learned Codebook. NeurIPS 2022.",
            "[5] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR 2015.",
            "[6] Deng, J., Guo, J., Ververas, E., Zafeiriou, S., & Deng, J. (2020). RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild. CVPR 2020.",
            "[7] Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017.",
            "[8] Replicate API Documentation — CodeFormer Model. https://replicate.com/sczhou/codeformer",
            "[9] PyTorch Documentation. https://pytorch.org/docs/",
            "[10] OpenCV Documentation. https://docs.opencv.org/",
        ]
        for ref in refs:
            self.p(ref)
        self.page_break()

    def build_bibliography(self):
        self.heading("BIBLIOGRAPHY", 1)
        bib = [
            "Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.",
            "Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O'Reilly Media.",
            "Flask Documentation. Pallets Projects. https://flask.palletsprojects.com/",
            "React Documentation. Meta Open Source. https://react.dev/",
            "CelebA-HQ Dataset. https://github.com/tkarras/progressive_growing_of_gans",
            "Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive Growing of GANs for Improved Quality, Stability, and Variation. ICLR 2018.",
            "Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.",
            "Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.",
            "IEEE Standard 830-1998: Recommended Practice for Software Requirements Specifications.",
            "Sommerville, I. (2016). Software Engineering (10th ed.). Pearson Education.",
        ]
        for entry in bib:
            self.p(entry)


def get_diagrams(use_attached=True):
    if use_attached and (ATTACHED_DIR / "architecture.png").exists():
        return {
            "architecture": attached_path("architecture.png"),
            "sequence": attached_path("sequence.png"),
            "database_schema": attached_path("database_schema.png"),
            "pipeline": attached_path("pipeline.png"),
            "project_structure": attached_path("project_structure.png"),
        }
    ensure_dirs()
    return {
        "architecture": draw_architecture_diagram(),
        "sequence": draw_sequence_diagram(),
        "database_schema": None,
        "pipeline": draw_pipeline_flow(),
        "project_structure": None,
    }


def generate_report_body(output_path=None, skip_cover=True, use_attached=True, add_page_template=False):
    """Generate report chapters (optionally without cover page)."""
    ensure_dirs()
    output_path = Path(output_path or DOCS_DIR / "report_body.pdf")
    diagrams = get_diagrams(use_attached=use_attached)
    arch_path = diagrams["architecture"]
    pipeline_path = diagrams["pipeline"]
    screenshots = collect_project_images()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2.2 * cm,
        leftMargin=2.2 * cm,
        topMargin=2.8 * cm,
        bottomMargin=2.2 * cm,
    )

    builder = ReportBuilder()
    if not skip_cover:
        builder.build_cover()
    builder.build_toc()
    builder.build_chapter1(arch_path)
    builder.build_chapter2(pipeline_path)
    builder.build_chapter3()
    builder.build_chapter4(diagrams)
    builder.build_chapter5(screenshots, diagrams)
    builder.build_chapter6()
    builder.build_conclusion()
    builder.build_future()
    builder.build_references()
    builder.build_bibliography()

    if add_page_template:
        doc.build(builder.story, onFirstPage=draw_page_template, onLaterPages=draw_page_template)
    else:
        doc.build(builder.story)
    return output_path


def main():
    path = generate_report_body(
        output_path=OUTPUT_PDF,
        skip_cover=False,
        use_attached=True,
    )
    print(f"\nReport generated: {path}")


if __name__ == "__main__":
    main()
