"""
Phase 1 PDF Report Generator.
Compiles audit results, exploratory figures, and problem formulation
into a formal academic PDF deliverable at reports/phase1.pdf.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUTPUT_PDF = os.path.join("reports", "phase1.pdf")
FIGURES_DIR = "figures"


def build_pdf():
    os.makedirs("reports", exist_ok=True)
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )

    story = []

    # Title Banner
    story.append(Paragraph("Phase 1 Report: Problem Understanding, Data Audit & EDA", title_style))
    story.append(Paragraph("<b>Course:</b> Data Analysis (Fall 1405) | <b>Track B:</b> Amazon Reviews '23 (Musical Instruments)", subtitle_style))
    story.append(Spacer(1, 10))

    # Section 1: Problem Formulation
    story.append(Paragraph("1. Problem Definition & Formulation", h1_style))
    story.append(Paragraph(
        "<b>Business Decision:</b> In the musical instruments marketplace, customer satisfaction is heavily influenced by the congruence between instrument specifications (price, brand tier) and user expectations articulated in reviews. This project formulates a multimodal predictive pipeline to assess customer satisfaction and predict review star ratings (1 to 5) by synergistically fusing review text representations with structured product metadata.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Core Research Question:</b> Does the integration of structured catalog metadata (price, store identity) with textual review semantics significantly reduce prediction error over unimodal text baselines while avoiding user/item data leakage?",
        body_style
    ))

    # Section 2: Data Audit Summary
    story.append(Paragraph("2. Data & Modality Availability Audit", h1_style))
    story.append(Paragraph(
        "We sampled 20,000 reviews and linked them against 11,423 unique parent instruments via HTTP streaming from the official UCSD repository with a 100% key-matching rate. Modality breakdown is detailed below:",
        body_style
    ))

    audit_data = [
        ["Modality Aspect", "Observations", "Coverage (%)"],
        ["Review Text (Non-empty)", "19,996", "99.98%"],
        ["Product Catalog Images", "19,993", "99.96%"],
        ["Store / Brand Name", "19,912", "99.56%"],
        ["Product Feature Bulletins", "18,900", "94.50%"],
        ["Valid Numeric Price", "13,287", "66.44%"],
        ["User-Uploaded Photos", "946", "4.73%"],
        ["Full Multimodal (Text + Img + Price)", "13,280", "66.40%"]
    ]
    t = Table(audit_data, colWidths=[240, 120, 120])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")])
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Section 3: Exploratory Data Analysis & Visualizations
    story.append(Paragraph("3. Exploratory Data Analysis (EDA) Insights", h1_style))
    story.append(Paragraph(
        "• <b>Severe Class Imbalance:</b> As shown in Figure 1, ratings exhibit strong positive skewness with 5-star ratings constituting over 65% of all feedback.<br/>"
        "• <b>Length vs. Sentiment Asymmetry:</b> Critical 1-star reviews exhibit notably longer mean word counts than satisfied 5-star reviews (Figure 2), indicating that dissatisfied musicians describe technical defects thoroughly.<br/>"
        "• <b>Modality Synergy:</b> Reviews containing user-uploaded images enjoy a significantly higher helpfulness rate compared to text-only submissions.",
        body_style
    ))

    # Attach figures
    fig1 = os.path.join(FIGURES_DIR, "eda_rating_distribution.png")
    fig2 = os.path.join(FIGURES_DIR, "eda_length_vs_rating.png")
    fig3 = os.path.join(FIGURES_DIR, "eda_modality_helpfulness.png")

    if os.path.exists(fig1):
        story.append(KeepTogether([
            Paragraph("<b>Figure 1:</b> Distribution of Star Ratings and Class Skewness", body_style),
            Image(fig1, width=420, height=220)
        ]))
        story.append(Spacer(1, 10))

    if os.path.exists(fig2):
        story.append(KeepTogether([
            Paragraph("<b>Figure 2:</b> Asymmetry between Review Length and Customer Rating", body_style),
            Image(fig2, width=420, height=230)
        ]))
        story.append(Spacer(1, 10))

    if os.path.exists(fig3):
        story.append(KeepTogether([
            Paragraph("<b>Figure 3:</b> Helpfulness Rate for Multi-Modal Reviews (User Images vs Text-only)", body_style),
            Image(fig3, width=420, height=230)
        ]))

    # Section 4: Leakage Prevention & Experimental Split
    story.append(Paragraph("4. Leakage Prevention & Experimental Design", h1_style))
    story.append(Paragraph(
        "<b>Split Strategy:</b> Standard random splitting introduces substantial data leakage because repeated interactions from identical users or identical products across folds allow models to memorize entity biases. To ensure authentic real-world generalization, an <b>Item-Disjoint Partition (70% Train, 15% Val, 15% Test)</b> is established, guaranteeing zero product overlap across sets.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Baselines:</b> Phase 3 benchmarks will evaluate (1) a naive Majority-Class Baseline, (2) an ordinary Least-Squares length heuristic, and (3) a text-only Ridge/TF-IDF regressor against multimodal fusion architectures.",
        body_style
    ))

    doc.build(story)
    print(f"[✓] Successfully compiled formal Phase 1 report to: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()