from __future__ import annotations

import os
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import formatdate
from email import encoders
from typing import List

logger = logging.getLogger(__name__)

class EmailSender:
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        subject_prefix: str = "PPE Compliance Report",
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.subject_prefix = subject_prefix

    def send_pdf(self, pdf_path: str, recipients: List[str], frame_count: int) -> None:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)
        if not self.username or not self.password:
            raise ValueError("SMTP_USERNAME / SMTP_PASSWORD not configured in .env")

        msg = MIMEMultipart()
        msg["From"] = self.username
        msg["To"] = ", ".join(recipients)
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = f"{self.subject_prefix} - {frame_count} events"

        body = (
            "Automated PPE compliance report attached.\n\n"
            f"Frames in report: {frame_count}\n"
            "\nThis email was sent from the Flask PPE monitoring system."
        )
        msg.attach(MIMEText(body, "plain"))

        filename = os.path.basename(pdf_path)
        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)

        logger.info("Connecting SMTP %s:%s", self.smtp_server, self.smtp_port)
        with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self.username, self.password)
            server.sendmail(self.username, recipients, msg.as_string())
        logger.info("✅ Email sent to %s", recipients)
