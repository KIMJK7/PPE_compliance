from __future__ import annotations

from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)

@pages_bp.get("/")
def index():
    return render_template("index.html")

@pages_bp.get("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@pages_bp.get("/live")
def live():
    return render_template("live_feed.html")

@pages_bp.route("/upload_detection")
def upload_detection():
    return render_template("upload_detection.html")
