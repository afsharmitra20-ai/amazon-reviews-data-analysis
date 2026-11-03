"""
Phase 1 Comprehensive PDF Report Generator.
Incorporates:
1. Problem Motivation & Business Understanding
2. Three Candidate Questions & Selection Justification (Explicitly required by guidelines)
3. Full Modality & Data Quality Audit
4. Exploratory Data Analysis & Visualizations
5. Detailed Data Leakage Audit (User, Item, Temporal) & Split Design
6. Explicit Baseline Definitions
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUTPUT_PDF = os.path.join("reports", "phase1.pdf")
FIGURES_DIR = "figures"


def build_pdf():
    os.makedirs("reports", exist_ok=True)
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=4
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading3"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=6,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=3
    )

    story = []

    # Title Banner
    story.append(Paragraph("Phase 1 Report: Problem Formulation, Modality Audit & EDA", title_style))
    story.append(Paragraph("<b>Course:</b> Data Analysis (Fall 1405) &nbsp;|&nbsp; <b>Track B:</b> Amazon Reviews '23 (Musical Instruments)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=8))

    # --- SECTION 1 ---
    story.append(Paragraph("1. Problem Understanding & Three Candidate Questions", h1_style))
    story.append(Paragraph(
        "In online musical instruments retail, customer purchase decisions are highly sensitive to qualitative technical feedback. We formulated three concrete research questions based on the problem context and initial data observations:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Question 1 (Multimodal Rating Prediction - Selected Primary Task):</b><br/>"
        "<i>Can the integration of textual review semantics with intrinsic product metadata (price, store brand, catalog imagery) predict fine-grained customer satisfaction and star ratings (1 to 5) with lower error than unimodal text models?</i><br/>"
        "<b>Practical Value:</b> Enables automated auditing of customer sentiment, calibration of rating inconsistencies, and fine-grained satisfaction scoring.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>Question 2 (Review Diagnostic Value / Helpfulness Prediction):</b><br/>"
        "<i>Can we predict whether a review will receive helpful votes based on technical vocabulary depth, length, reviewer purchase verification, and product price?</i><br/>"
        "<b>Limitation / Rejection Reason:</b> As audited in our EDA, 'helpful_vote' suffers from severe temporal exposure bias (older reviews accumulate disproportionately more votes irrespective of intrinsic quality), introducing uncontrolled temporal confounding.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>Question 3 (Rating-Review Sentiment Inconsistency Detection):</b><br/>"
        "<i>Can we identify anomalous interactions where review text sentiment contradicts the assigned numerical star rating (e.g., highly negative text with 5 stars)?</i><br/>"
        "<b>Limitation / Rejection Reason:</b> While valuable for fraud auditing, it represents a sparse anomaly detection setup with no direct ground-truth labels, making benchmark evaluation subjective.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>Justification for Selecting Question 1:</b> Question 1 directly leverages the core multimodal nature of Track B (Text + Structured + Image). It operates on 100% available ground truth ('rating'), provides substantial real-world decision utility for catalog managers, and allows rigorous fusion vs. unimodal ablation testing in Phases 3 and 4.",
        body_style
    ))

    # --- SECTION 2 ---
    story.append(Paragraph("2. Data & Modality Availability Audit", h1_style))
    story.append(Paragraph(
        "We collected a representative sample of 20,000 review interactions and matched them against 11,423 parent products with a <b>100.00% foreign-key match rate</b>. Modality coverage metrics are summarized below:",
        body_style
    ))

    audit_data = [
        ["Modality Component", "Observations", "Coverage (%)"],
        ["Review Text (Non-empty)", "19,996", "99.98%"],
        ["Review Title (Non-empty)", "20,000", "100.00%"],
        ["Product Catalog Images", "19,993", "99.96%"],
        ["Brand / Store Identity", "19,912", "99.56%"],
        ["Product Feature Bulletins", "18,900", "94.50%"],
        ["Valid Numeric Price", "13,287", "66.44%"],
        ["User-Uploaded Photos", "946", "4.73%"],
        ["Full Multimodal (Text + Img + Price)", "13,280", "66.40%"]
    ]
    t = Table(audit_data, colWidths=[240, 110, 110])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")])
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # --- SECTION 3 ---
    story.append(Paragraph("3. Exploratory Data Analysis (EDA) Insights", h1_style))
    story.append(Paragraph(
        "• <b>Class Imbalance & Positivity Bias:</b> Over 65% of ratings are 5-star, creating substantial class imbalance that requires stratified validation and macro-averaged metrics.<br/>"
        "• <b>Text Length vs. Sentiment Asymmetry:</b> Critical 1-star reviews exhibit longer average word counts than satisfied 5-star reviews, reflecting detailed problem explanations by disappointed musicians.<br/>"
        "• <b>Modality Helpfulness Effect:</b> User-uploaded photos occur in only 4.73% of reviews but correlate with a substantially higher helpfulness rate.",
        body_style
    ))

    fig1 = os.path.join(FIGURES_DIR, "eda_rating_distribution.png")
    fig2 = os.path.join(FIGURES_DIR, "eda_length_vs_rating.png")
    fig3 = os.path.join(FIGURES_DIR, "eda_modality_helpfulness.png")

    if os.path.exists(fig1):
        story.append(KeepTogether([
            Paragraph("<b>Figure 1:</b> Distribution of Star Ratings (Imbalance Audit)", h2_style),
            Image(fig1, width=400, height=195)
        ]))

    if os.path.exists(fig2):
        story.append(KeepTogether([
            Paragraph("<b>Figure 2:</b> Asymmetry between Review Word Count and Rating", h2_style),
            Image(fig2, width=400, height=200)
        ]))

    if os.path.exists(fig3):
        story.append(KeepTogether([
            Paragraph("<b>Figure 3:</b> Helpfulness Rate for Multi-Modal Reviews", h2_style),
            Image(fig3, width=400, height=200)
        ]))

    # --- SECTION 4 ---
    story.append(Paragraph("4. Data Leakage Audit & Experimental Split Design", h1_style))
    story.append(Paragraph(
        "A conventional <i>Random Row Split</i> is scientifically invalid for this problem due to three distinct leakage vectors:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. User-Level Leakage:</b> A reviewer evaluating multiple instruments shares identical vocabulary patterns and rating habits. Overlapping users between splits leads to overly optimistic evaluation.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>2. Item / Cold-Start Leakage:</b> If reviews of the same instrument appear in both training and test sets, the model simply memorizes product-level average ratings rather than learning semantic sentiment.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>3. Temporal Leakage:</b> Utilizing future product feedback to predict past reviews violates chronological causality.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>Adopted Partition Strategy:</b> We implement an <b>Item-Disjoint Partition (70% Train, 15% Validation, 15% Test)</b> grouped strictly on <code>parent_asin</code> with fixed seed 42. Unseen products in the test set guarantee an authentic cold-start evaluation.",
        body_style
    ))

    # --- SECTION 5 ---
    story.append(Paragraph("5. Baseline Definitions", h1_style))
    story.append(Paragraph(
        "To rigorously prove the value-add of multimodal modeling in subsequent phases, three explicit baselines are established:<br/>"
        "• <b>Baseline 1 (Majority Class / Mode):</b> Predicts the dominant 5-star rating for all instances (benchmarking class imbalance resilience).<br/>"
        "• <b>Baseline 2 (Heuristic Length / Tabular OLS):</b> Simple linear regression using review word count and log-transformed product price.<br/>"
        "• <b>Baseline 3 (Unimodal Text TF-IDF + Ridge):</b> Standard text-only linear model with n-gram TF-IDF representations, serving as the benchmark to measure multimodal gain.",
        body_style
    ))

    doc.build(story)
    print(f"[✓] Successfully recompiled comprehensive Phase 1 PDF report to: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()