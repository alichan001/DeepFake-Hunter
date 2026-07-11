"""
PDF Report Generator
====================
Generates a professional forensic-style PDF report from a deepfake
analysis result dict.

Usage (from Flask route):
    from pdf_report import generate_pdf_report
    pdf_bytes = generate_pdf_report(result_dict, video_filename)

Dependencies:
    pip install reportlab matplotlib

No other new dependencies — matplotlib is already in your requirements.txt.
"""

import io
import math
import datetime
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes   import A4
from reportlab.lib.units       import mm, cm
from reportlab.lib             import colors
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums       import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, KeepTogether
)
from reportlab.platypus        import PageBreak


# ── brand colours ─────────────────────────────────────────────────────────────
C_BG        = colors.HexColor("#080a10")
C_SURFACE   = colors.HexColor("#111520")
C_SURFACE2  = colors.HexColor("#181d2a")
C_BLUE      = colors.HexColor("#3b82f6")
C_BLUE_LT   = colors.HexColor("#60a5fa")
C_GREEN     = colors.HexColor("#22c55e")
C_RED       = colors.HexColor("#ef4444")
C_AMBER     = colors.HexColor("#f59e0b")
C_PURPLE    = colors.HexColor("#8b5cf6")
C_TEXT      = colors.HexColor("#e2e6f0")
C_TEXT2     = colors.HexColor("#a0a8be")
C_MUTED     = colors.HexColor("#5c6480")
C_BORDER    = colors.HexColor("#1f2535")
C_WHITE     = colors.white
C_BLACK     = colors.black

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ── matplotlib chart helpers ───────────────────────────────────────────────────

def _chart_style(ax):
    """Apply dark theme to a matplotlib axes."""
    ax.set_facecolor("#111520")
    ax.figure.patch.set_facecolor("#0d1018")
    ax.tick_params(colors="#5c6480", labelsize=7)
    ax.spines["bottom"].set_color("#1f2535")
    ax.spines["left"].set_color("#1f2535")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#5c6480")
    ax.xaxis.label.set_color("#5c6480")
    ax.grid(color="#1f2535", linewidth=0.5)


def _fig_to_image(fig, width_mm: float, height_mm: float) -> Image:
    """Convert a matplotlib figure to a ReportLab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_mm * mm, height=height_mm * mm)


def _blink_chart(blink_timeline: List[int], width_mm=82, height_mm=38) -> Image:
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    _chart_style(ax)
    x = list(range(len(blink_timeline)))
    labels = [f"{i*5}s" for i in x]
    ax.bar(x, blink_timeline, color="#3b82f6", alpha=0.85, width=0.6)
    ax.set_xticks(x[::max(1, len(x)//6)])
    ax.set_xticklabels(labels[::max(1, len(x)//6)], rotation=0, ha="center")
    ax.set_ylabel("Blinks", fontsize=7, color="#5c6480")
    ax.set_title("Blink Timeline (per 5 s window)", fontsize=8,
                 color="#a0a8be", pad=6)
    fig.tight_layout(pad=0.4)
    return _fig_to_image(fig, width_mm, height_mm)


def _rppg_chart(waveform: List[float], has_heartbeat: bool,
                width_mm=160, height_mm=38) -> Image:
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    _chart_style(ax)
    clr = "#22c55e" if has_heartbeat else "#5c6480"
    ax.plot(waveform, color=clr, linewidth=0.9, alpha=0.9)
    ax.set_ylabel("Amplitude", fontsize=7, color="#5c6480")
    ax.set_xlabel("Frame", fontsize=7, color="#5c6480")
    lbl = "Heartbeat signal detected" if has_heartbeat else "No heartbeat signal"
    ax.set_title(f"rPPG Signal — {lbl}", fontsize=8, color="#a0a8be", pad=6)
    fig.tight_layout(pad=0.4)
    return _fig_to_image(fig, width_mm, height_mm)


def _eye_chart(gaze_x: List[float], width_mm=82, height_mm=38) -> Image:
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    _chart_style(ax)
    ax.plot(gaze_x, color="#f59e0b", linewidth=0.9, alpha=0.9)
    ax.set_ylabel("Gaze X (px)", fontsize=7, color="#5c6480")
    ax.set_title("Eye Movement (Horizontal Gaze)", fontsize=8,
                 color="#a0a8be", pad=6)
    fig.tight_layout(pad=0.4)
    return _fig_to_image(fig, width_mm, height_mm)


def _radar_chart(scores: Dict[str, float], width_mm=75, height_mm=75) -> Image:
    """
    Radar chart showing all signal scores (0-100, higher = more real).
    """
    labels = list(scores.keys())
    vals   = [max(0, min(100, v)) for v in scores.values()]
    N      = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    vals_plot = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4),
                           subplot_kw=dict(polar=True))
    ax.set_facecolor("#111520")
    fig.patch.set_facecolor("#0d1018")

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color="#5c6480", fontsize=6)
    ax.yaxis.grid(color="#1f2535", linewidth=0.6)
    ax.xaxis.grid(color="#1f2535", linewidth=0.6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#a0a8be", fontsize=6.5)
    ax.spines["polar"].set_color("#1f2535")

    # Fill
    ax.fill(angles, vals_plot, color="#3b82f6", alpha=0.18)
    ax.plot(angles, vals_plot, color="#60a5fa", linewidth=1.5)
    ax.scatter(angles[:-1], vals, color="#60a5fa", s=20, zorder=5)

    fig.tight_layout(pad=0.3)
    return _fig_to_image(fig, width_mm, height_mm)


# ── reportlab styles ──────────────────────────────────────────────────────────

def _make_styles():
    base = getSampleStyleSheet()

    def ps(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "title": ps("RTitle",
            fontSize=22, fontName="Helvetica-Bold",
            textColor=C_TEXT, alignment=TA_LEFT, spaceAfter=2),

        "subtitle": ps("RSubtitle",
            fontSize=11, fontName="Helvetica",
            textColor=C_TEXT2, alignment=TA_LEFT, spaceAfter=0),

        "section": ps("RSection",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=C_MUTED, alignment=TA_LEFT,
            spaceBefore=10, spaceAfter=6,
            textTransform="uppercase", letterSpacing=1.2),

        "body": ps("RBody",
            fontSize=9, fontName="Helvetica",
            textColor=C_TEXT2, leading=14, spaceAfter=4),

        "verdict_fake": ps("RVFake",
            fontSize=28, fontName="Helvetica-Bold",
            textColor=C_RED, alignment=TA_LEFT),

        "verdict_real": ps("RVReal",
            fontSize=28, fontName="Helvetica-Bold",
            textColor=C_GREEN, alignment=TA_LEFT),

        "small": ps("RSmall",
            fontSize=7.5, fontName="Helvetica",
            textColor=C_MUTED, leading=11),

        "mono": ps("RMono",
            fontSize=8, fontName="Courier",
            textColor=C_TEXT2, leading=12),

        "caption": ps("RCaption",
            fontSize=7, fontName="Helvetica",
            textColor=C_MUTED, alignment=TA_CENTER),

        "label": ps("RLabel",
            fontSize=7.5, fontName="Helvetica-Bold",
            textColor=C_MUTED, spaceAfter=1,
            textTransform="uppercase"),

        "value": ps("RValue",
            fontSize=12, fontName="Helvetica-Bold",
            textColor=C_TEXT, spaceAfter=6),
    }


# ── colour helpers ────────────────────────────────────────────────────────────

def _status_color(status: str) -> colors.Color:
    return {
        "ok":      C_GREEN,
        "warning": C_AMBER,
        "danger":  C_RED,
    }.get(status, C_TEXT2)


def _verdict_color(verdict: str) -> colors.Color:
    return C_RED if verdict == "FAKE" else C_GREEN


# ── page background ───────────────────────────────────────────────────────────

def _draw_bg(canvas, doc):
    """Draw dark background and top bar on every page."""
    canvas.saveState()
    # Full page dark background
    canvas.setFillColor(C_BG)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Top accent bar (gradient-like using two rects)
    bar_h = 6
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, PAGE_H - bar_h, PAGE_W * 0.6, bar_h, fill=1, stroke=0)
    canvas.setFillColor(C_PURPLE)
    canvas.rect(PAGE_W * 0.6, PAGE_H - bar_h, PAGE_W * 0.4, bar_h, fill=1, stroke=0)

    # Footer
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(MARGIN, 10 * mm,
                      f"DeepGuard Forensic Report — Page {doc.page} — CONFIDENTIAL")
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm,
                           "Generated by DeepGuard FYP System")
    canvas.restoreState()


# ── dark table style ──────────────────────────────────────────────────────────

def _dark_table(col_widths, data, header=True) -> Table:
    tbl = Table(data, colWidths=col_widths)
    cmds = [
        ("BACKGROUND",  (0, 0), (-1, -1), C_SURFACE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_SURFACE, C_SURFACE2]),
        ("TEXTCOLOR",   (0, 0), (-1, -1), C_TEXT2),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.4, C_BORDER),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]
    if header:
        cmds += [
            ("BACKGROUND",  (0, 0), (-1, 0), C_SURFACE2),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 7.5),
            ("TEXTCOLOR",   (0, 0), (-1, 0), C_MUTED),
        ]
    tbl.setStyle(TableStyle(cmds))
    return tbl


# ── coloured pill cell ────────────────────────────────────────────────────────

def _pill(text: str, status: str, styles) -> Paragraph:
    clr = {
        "ok":      "#22c55e",
        "warning": "#f59e0b",
        "danger":  "#ef4444",
    }.get(status, "#5c6480")
    return Paragraph(
        f'<font color="{clr}"><b>{text}</b></font>',
        styles["small"]
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(
    result: Dict[str, Any],
    video_filename: str = "video.mp4"
) -> bytes:
    """
    Generate a PDF forensic report from a deepfake analysis result.

    Parameters
    ----------
    result         : dict returned by analyze_video() / combine_signals()
    video_filename : original video filename for the report header

    Returns
    -------
    bytes — raw PDF content ready to send as HTTP response
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 8, bottomMargin=20 * mm,
        title="DeepGuard Forensic Report",
        author="DeepGuard FYP System",
    )

    S       = _make_styles()
    story   = []
    W       = PAGE_W - 2 * MARGIN          # usable width

    verdict    = result.get("verdict",    "UNKNOWN")
    confidence = result.get("confidence", 0)
    score      = result.get("score",      0)
    explanation= result.get("explanation","—")
    evidence   = result.get("evidence",   [])
    signals    = result.get("signals",    {})
    meta       = result.get("meta",       {})
    is_fake    = verdict == "FAKE"
    now        = datetime.datetime.now().strftime("%d %B %Y, %H:%M:%S")

    # ── PAGE 1 ── HEADER ─────────────────────────────────────────────────────
    story.append(Spacer(1, 4 * mm))

    # Logo row
    logo_data = [[
        Paragraph("<b>🛡 DeepGuard</b>", ParagraphStyle(
            "logo", fontSize=16, fontName="Helvetica-Bold",
            textColor=C_BLUE_LT)),
        Paragraph(
            f'<font color="#5c6480">Forensic Analysis Report<br/>'
            f'{now}</font>',
            ParagraphStyle("logoR", fontSize=8, fontName="Helvetica",
                           textColor=C_MUTED, alignment=TA_RIGHT))
    ]]
    logo_tbl = Table(logo_data, colWidths=[W * 0.5, W * 0.5])
    logo_tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    story.append(logo_tbl)
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 5 * mm))

    # ── VERDICT HERO ─────────────────────────────────────────────────────────
    v_color = C_RED if is_fake else C_GREEN
    v_bg    = colors.HexColor("#1c0d0d") if is_fake else colors.HexColor("#0b1c10")
    v_icon  = "⚠" if is_fake else "✓"

    verdict_data = [[
        Paragraph(
            f'<font color="{"#ef4444" if is_fake else "#22c55e"}">'
            f'<b>{v_icon}  {verdict}</b></font>',
            ParagraphStyle("verd", fontSize=26, fontName="Helvetica-Bold",
                           textColor=v_color)),
        Paragraph(
            f'<font color="#a0a8be">Confidence Score</font><br/>'
            f'<font color="{"#ef4444" if is_fake else "#22c55e"}" size="20">'
            f'<b>{confidence}%</b></font>',
            ParagraphStyle("conf", fontSize=10, fontName="Helvetica",
                           textColor=C_TEXT2, alignment=TA_RIGHT)),
    ]]
    v_tbl = Table(verdict_data, colWidths=[W * 0.6, W * 0.4])
    v_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), v_bg),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), [6,6,6,6]),
        ("LEFTPADDING",  (0,0), (-1,-1), 14),
        ("RIGHTPADDING", (0,0), (-1,-1), 14),
        ("TOPPADDING",   (0,0), (-1,-1), 12),
        ("BOTTOMPADDING",(0,0), (-1,-1), 12),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("BOX",          (0,0), (-1,-1), 0.8, v_color),
    ]))
    story.append(v_tbl)
    story.append(Spacer(1, 3 * mm))

    # Explanation box
    exp_data = [[
        Paragraph('<font color="#5c6480"><b>WHY THIS VERDICT?</b></font>',
                  ParagraphStyle("expl_lbl", fontSize=7, fontName="Helvetica-Bold",
                                 textColor=C_MUTED, leading=10)),
    ],[
        Paragraph(f'<font color="#a0a8be">{explanation}</font>',
                  ParagraphStyle("expl", fontSize=9, fontName="Helvetica",
                                 textColor=C_TEXT2, leading=13)),
    ]]
    exp_tbl = Table(exp_data, colWidths=[W])
    exp_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), C_SURFACE),
        ("LEFTPADDING",  (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("BOX",          (0,0), (-1,-1), 0.4, C_BORDER),
    ]))
    story.append(exp_tbl)
    story.append(Spacer(1, 5 * mm))

    # ── VIDEO METADATA ────────────────────────────────────────────────────────
    story.append(Paragraph("VIDEO METADATA", S["section"]))

    meta_rows = [
        ["FIELD", "VALUE"],
        ["Filename",          video_filename],
        ["Verdict",           verdict],
        ["Raw Score",         f"{score:.4f}  (threshold: 0.45)"],
        ["FPS",               f"{meta.get('fps', 0):.1f}"],
        ["Total Frames",      str(meta.get("total_frames", "—"))],
        ["Analyzed Frames",   str(meta.get("analyzed_frames", "—"))],
        ["Face Detection",    f"{meta.get('face_detection_rate', 0) * 100:.1f}%"],
        ["Frame Skip",        str(meta.get("frame_skip", "—"))],
        ["Processing Time",   f"{meta.get('processing_time_sec', '—')} seconds"],
        ["Adaptive EAR",      str(meta.get("adaptive_ear_threshold", "—"))],
        ["Report Generated",  now],
    ]
    meta_tbl = _dark_table(
        [W * 0.35, W * 0.65],
        [[Paragraph(c, S["small"]) for c in row] for row in meta_rows]
    )
    story.append(meta_tbl)
    story.append(Spacer(1, 5 * mm))

    # ── SIGNAL BREAKDOWN TABLE ────────────────────────────────────────────────
    story.append(Paragraph("SIGNAL BREAKDOWN — 6 ANALYZERS", S["section"]))

    sig_rows = [["SIGNAL", "VALUE", "STATUS", "ANALYSIS"]]
    for ev in evidence:
        sig_rows.append([
            Paragraph(f'<b>{ev.get("signal","")}</b>', S["small"]),
            Paragraph(ev.get("value", "—"),            S["mono"]),
            _pill(ev.get("status","").upper(), ev.get("status",""), S),
            Paragraph(ev.get("message",""),            S["small"]),
        ])

    sig_tbl = _dark_table(
        [W * 0.20, W * 0.18, W * 0.12, W * 0.50],
        sig_rows
    )
    # Colour status column per row
    for i, ev in enumerate(evidence, start=1):
        clr = _status_color(ev.get("status",""))
        sig_tbl.setStyle(TableStyle([
            ("TEXTCOLOR", (2, i), (2, i), clr),
            ("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"),
        ]))
    story.append(sig_tbl)
    story.append(Spacer(1, 5 * mm))

    # ── RADAR CHART + SCORE GRID ──────────────────────────────────────────────
    story.append(Paragraph("SIGNAL SCORES — PHYSIOLOGICAL AUTHENTICITY", S["section"]))

    # Build radar scores (0=fake, 100=real for each signal)
    blink_rate = signals.get("blink_rate") or 0
    blink_score = max(0, min(100,
        100 if 10 <= blink_rate <= 25
        else max(0, 100 - abs(blink_rate - 17) * 8)))

    eye_ent   = signals.get("eye_entropy") or 0
    eye_score = min(100, eye_ent / 4.0 * 100)

    hb        = signals.get("has_heartbeat", False)
    hr_snr    = signals.get("rppg_snr") or 0
    rppg_score= min(100, hr_snr / 5.0 * 100) if hb else 0

    ms_ratio  = signals.get("microsaccade_ratio") or 0
    ms_score  = min(100, ms_ratio / 0.5 * 100)

    fou_art   = signals.get("fourier_artifact") or 0
    fou_score = max(0, 100 - fou_art * 200)

    geo_stab  = signals.get("geom_stability") or 0
    geo_score = min(100, geo_stab * 100)

    radar_scores = {
        "Blink":     round(blink_score),
        "Eye Move":  round(eye_score),
        "rPPG":      round(rppg_score),
        "Microsacc": round(ms_score),
        "Fourier":   round(fou_score),
        "Geometry":  round(geo_score),
    }

    radar_img = _radar_chart(radar_scores, width_mm=70, height_mm=70)

    # Score cells table
    score_data_header = [["SIGNAL", "SCORE", "INDICATOR"]]
    for name, val in radar_scores.items():
        bar_full = "█" * int(val / 10) + "░" * (10 - int(val / 10))
        status   = "ok" if val >= 60 else ("warning" if val >= 35 else "danger")
        clr      = "#22c55e" if val >= 60 else ("#f59e0b" if val >= 35 else "#ef4444")
        score_data_header.append([
            Paragraph(name, S["small"]),
            Paragraph(f'<font color="{clr}"><b>{val}</b></font>', S["small"]),
            Paragraph(f'<font color="{clr}" face="Courier">{bar_full}</font>',
                      S["small"]),
        ])

    scores_tbl = _dark_table(
        [W * 0.30, W * 0.15, W * 0.55],
        score_data_header
    )

    radar_cell = [[radar_img, scores_tbl]]
    combined   = Table(radar_cell, colWidths=[72 * mm, W - 72 * mm])
    combined.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    story.append(combined)
    story.append(Spacer(1, 5 * mm))

    # ── PAGE 2 — CHARTS ──────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("SIGNAL WAVEFORMS & CHARTS", S["section"]))

    chart_w_half = (W / 2 - 3 * mm)

    # Row 1: blink + eye side by side
    blink_tl  = signals.get("blink_timeline", [])
    gaze_x    = signals.get("gaze_x_series",  [])

    left_col  = []
    right_col = []

    if blink_tl:
        left_col.append(_blink_chart(blink_tl, width_mm=chart_w_half / mm, height_mm=42))
        left_col.append(Paragraph("Blink counts per 5-second window. Low counts may indicate synthetic generation.", S["caption"]))
    else:
        left_col.append(Paragraph("No blink data.", S["small"]))

    if gaze_x:
        right_col.append(_eye_chart(gaze_x, width_mm=chart_w_half / mm, height_mm=42))
        right_col.append(Paragraph("Horizontal gaze displacement. Natural eyes show irregular saccadic motion.", S["caption"]))
    else:
        right_col.append(Paragraph("No gaze data.", S["small"]))

    c_tbl = Table([[left_col, right_col]], colWidths=[chart_w_half, chart_w_half])
    c_tbl.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 3),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(c_tbl)
    story.append(Spacer(1, 4 * mm))

    # Row 2: rPPG full width
    rppg_wave = signals.get("rppg_waveform", [])
    has_hb    = signals.get("has_heartbeat", False)
    if rppg_wave:
        story.append(_rppg_chart(rppg_wave, has_hb, width_mm=W / mm, height_mm=42))
        story.append(Paragraph(
            "Remote Photoplethysmography (rPPG) signal extracted from facial "
            "skin color variation. A genuine heartbeat produces a periodic "
            "waveform. Absence of periodicity is a strong indicator of deepfake.",
            S["caption"]))
        story.append(Spacer(1, 5 * mm))

    # ── DETAILED NUMERIC DATA ─────────────────────────────────────────────────
    story.append(Paragraph("DETAILED NUMERIC DATA", S["section"]))

    num_rows = [["METRIC", "VALUE", "METRIC", "VALUE"]]

    def fmt(v):
        if v is None: return "—"
        if isinstance(v, float): return f"{v:.4f}"
        return str(v)

    flat = [
        ("Blink Rate",           fmt(signals.get("blink_rate"))),
        ("Blink Count",          fmt(signals.get("blink_count"))),
        ("Eye Entropy",          fmt(signals.get("eye_entropy"))),
        ("Eye Vel Std",          fmt(signals.get("eye_velocity_std"))),
        ("Heart Rate (bpm)",     fmt(signals.get("heart_rate"))),
        ("Heartbeat Detected",   str(signals.get("has_heartbeat", "—"))),
        ("rPPG SNR",             fmt(signals.get("rppg_snr"))),
        ("Microsaccade Ratio",   fmt(signals.get("microsaccade_ratio"))),
        ("Micro Event Rate",     fmt(signals.get("micro_event_rate"))),
        ("Fourier Artifact",     fmt(signals.get("fourier_artifact"))),
        ("Fourier Deviation",    fmt(signals.get("fourier_deviation"))),
        ("Geometry Mean CV",     fmt(signals.get("geom_mean_cv"))),
        ("Geometry Stability",   fmt(signals.get("geom_stability"))),
        ("Raw Fake Score",       fmt(score)),
    ]

    # Pair them into rows of 4 cols
    paired = []
    for i in range(0, len(flat), 2):
        r1 = flat[i]
        r2 = flat[i+1] if i+1 < len(flat) else ("", "")
        paired.append([
            Paragraph(r1[0], S["small"]),
            Paragraph(r1[1], S["mono"]),
            Paragraph(r2[0], S["small"]),
            Paragraph(r2[1], S["mono"]),
        ])

    num_tbl = _dark_table(
        [W*0.28, W*0.22, W*0.28, W*0.22],
        [["METRIC","VALUE","METRIC","VALUE"]] + paired
    )
    story.append(num_tbl)
    story.append(Spacer(1, 5 * mm))

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.4, color=C_BORDER))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "DISCLAIMER: This report was generated automatically by the DeepGuard "
        "physiological signal analysis system. Results are based on blink rate, "
        "eye movement entropy, rPPG heartbeat, microsaccade frequency, Fourier "
        "frequency artifacts, and facial geometry consistency. This tool is intended "
        "for research and educational purposes. Results should be interpreted by a "
        "qualified professional and should not be used as sole evidence in legal "
        "proceedings. Final Year Project — University Research.",
        S["small"]
    ))

    # ── BUILD ─────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_draw_bg, onLaterPages=_draw_bg)
    return buf.getvalue()
