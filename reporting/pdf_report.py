from __future__ import annotations

import os
import io
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    def __init__(self, reports_dir: str):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate(
        self,
        payload: Dict[str, Any],
        title: str = "PPE Compliance Report",
    ) -> str:
        """Generate report from FrameStore payload:
        payload = {total, compliant:[FrameRecord dict], non_compliant:[...]}
        Returns path to PDF.
        """
        now = datetime.now()
        filename = f"compliance_report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
        path = os.path.join(self.reports_dir, filename)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=22,
            textColor=colors.HexColor("#1f77b4"),
            alignment=TA_CENTER,
            spaceAfter=18,
        )
        meta_style = ParagraphStyle(
            "Meta",
            parent=styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#666666"),
            alignment=TA_LEFT,
        )

        total = int(payload.get("total", 0))
        compliant = [f for f in (payload.get("compliant", []) or [])
                     if float(f.get("avg_confidence", 0.0) or 0.0) >= 0.25]

        non_compliant = [f for f in (payload.get("non_compliant", []) or [])
                         if float(f.get("avg_confidence", 0.0) or 0.0) >= 0.25]

        rate = (len(compliant) / total * 100.0) if total else 0.0

        doc = SimpleDocTemplate(path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story: List[Any] = []
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"<b>Generated:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
        story.append(Paragraph(f"<b>Total Frames:</b> {total}", meta_style))
        story.append(Paragraph(f"<b>Compliant:</b> {len(compliant)}", meta_style))
        story.append(Paragraph(f"<b>Non-Compliant:</b> {len(non_compliant)}", meta_style))
        story.append(Paragraph(f"<b>Compliance Rate:</b> {rate:.1f}%", meta_style))
        story.append(Spacer(1, 0.2 * inch))

        # Executive summary
        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(
            "This report was generated manually from the live PPE monitoring dashboard. "
            "Non-compliant frames are listed first for quick review.",
            styles["BodyText"],
        ))
        story.append(Spacer(1, 0.2 * inch))

        def add_section(section_title: str, frames: List[Dict[str, Any]], priority: bool) -> None:
            story.append(Paragraph(section_title, styles["Heading2"]))
            story.append(Spacer(1, 0.1 * inch))

            if not frames:
                story.append(Paragraph("No frames captured in this section.", styles["BodyText"]))
                story.append(Spacer(1, 0.2 * inch))
                return

            # Table header
            table_data = [["#", "Timestamp", "Reason", "Classes", "Avg Conf", "Frame"]]
            for idx, fr in enumerate(frames, 1):
                img_obj = _data_url_to_rl_image(fr.get("image_data", ""), width=2.2*inch, height=1.4*inch)
                classes = ", ".join(fr.get("classes") or [])
                table_data.append([
                    str(idx),
                    fr.get("timestamp", ""),
                    fr.get("reason", ""),
                    classes or "-",
                    f"{float(fr.get('avg_confidence', 0.0))*100:.1f}%",
                    img_obj or "(image unavailable)",
                ])

                # keep pages manageable
                if idx % 6 == 0:
                    t = Table(table_data, colWidths=[0.3*inch, 1.2*inch, 1.2*inch, 1.3*inch, 0.7*inch, 2.3*inch])
                    t.setStyle(_default_table_style(priority=priority))
                    story.append(t)
                    story.append(PageBreak())
                    table_data = [["#", "Timestamp", "Reason", "Classes", "Avg Conf", "Frame"]]

            t = Table(table_data, colWidths=[0.3*inch, 1.2*inch, 1.2*inch, 1.3*inch, 0.7*inch, 2.3*inch])
            t.setStyle(_default_table_style(priority=priority))
            story.append(t)
            story.append(Spacer(1, 0.2 * inch))

        # Non-compliant first
        add_section("Non-Compliant Frames", non_compliant, priority=True)
        add_section("Compliant Frames", compliant, priority=False)

        doc.build(story)
        logger.info("✅ PDF report generated: %s", path)
        return path

def _default_table_style(priority: bool) -> TableStyle:
    header_bg = colors.HexColor("#ef4444") if priority else colors.HexColor("#22c55e")
    return TableStyle([
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,1), (-1,-1), 8),
    ])

def _data_url_to_rl_image(data_url: str, width: float, height: float):
    try:
        if not data_url.startswith("data:image"):
            return None
        b64 = data_url.split(",", 1)[1]
        raw = base64.b64decode(b64)
        pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
        out = io.BytesIO()
        pil.save(out, format="PNG")
        out.seek(0)
        return Image(out, width=width, height=height)
    except Exception:
        return None
