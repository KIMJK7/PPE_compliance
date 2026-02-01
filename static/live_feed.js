// ===============================
// Live Feed Page Logic
// ===============================

// Configuration
const STATUS_CHECK_INTERVAL = 1000; // Check status every 1 second for live updates
const COMPLIANCE_CHECK_INTERVAL = 500; // Check compliance every 0.5 seconds

// State
let statusTimer = null;
let complianceTimer = null;
let stats = {
  totalFrames: 0,
  framesWithPerson: 0,
  compliantFrames: 0,
  nonCompliantFrames: 0,
};

// ===============================
// Initialization
// ===============================
document.addEventListener("DOMContentLoaded", () => {
  console.log("Live feed page loaded");

  // Setup event listeners
  setupEventListeners();

  // Start polling for updates
  startStatusPolling();
  startCompliancePolling();

  // Initial check
  checkStreamStatus();
});

// ===============================
// Event Listeners
// ===============================
function setupEventListeners() {
  // Refresh button
  const refreshBtn = document.getElementById("refreshBtn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", refreshStream);
  }

  // Video stream events
  const videoStream = document.getElementById("videoStream");
  if (videoStream) {
    videoStream.addEventListener("load", () => {
      console.log("Stream loaded successfully");
      updateStreamInfo("Stream active", "success");
    });

    videoStream.addEventListener("error", () => {
      console.error("Stream failed to load");
      updateStreamInfo(
        "Stream error - check if RTSP camera is connected",
        "error",
      );
    });
  }
}

// ===============================
// Status Polling
// ===============================
function startStatusPolling() {
  // Clear existing timer
  if (statusTimer) {
    clearInterval(statusTimer);
  }

  // Poll for status updates
  statusTimer = setInterval(checkStreamStatus, STATUS_CHECK_INTERVAL);
}

function startCompliancePolling() {
  // Clear existing timer
  if (complianceTimer) {
    clearInterval(complianceTimer);
  }

  // Poll for compliance updates
  complianceTimer = setInterval(
    checkComplianceStatus,
    COMPLIANCE_CHECK_INTERVAL,
  );
}

// ===============================
// API Calls
// ===============================
async function checkStreamStatus() {
  try {
    const response = await fetch("/api/cctv/status");
    const data = await response.json();

    // Update stats
    if (data.person_detection) {
      const pd = data.person_detection;
      stats.totalFrames = pd.total_frames_processed || 0;
      stats.framesWithPerson = pd.frames_with_person || 0;

      // Update UI
      updateStatusUI(data);
    }
  } catch (error) {
    console.error("Failed to fetch stream status:", error);
  }
}

async function checkComplianceStatus() {
  try {
    const response = await fetch("/api/cctv/compliance");

    if (response.status === 503) {
      // No compliance data yet
      updateComplianceBadge("WAITING", "waiting", "â³");
      return;
    }

    const data = await response.json();

    // Update compliance UI
    updateComplianceUI(data);

    // Update stats
    if (data.is_compliant) {
      stats.compliantFrames++;
    } else {
      stats.nonCompliantFrames++;
    }
  } catch (error) {
    console.error("Failed to fetch compliance status:", error);
  }
}

// ===============================
// UI Updates
// ===============================
function updateStatusUI(data) {
  // Update stream info
  if (data.stream_active) {
    updateStreamInfo(
      `Stream active Â· ${data.frame_dimensions?.width}x${data.frame_dimensions?.height}`,
      "success",
    );
  } else {
    updateStreamInfo("Stream inactive", "error");
  }

  // Update person detection stats
  const personDetectionRate = document.getElementById("personDetectionRate");
  if (personDetectionRate && data.person_detection) {
    personDetectionRate.textContent =
      data.person_detection.person_detection_rate;
  }

  const framesProcessed = document.getElementById("framesProcessed");
  if (framesProcessed && data.person_detection) {
    framesProcessed.textContent =
      data.person_detection.total_frames_processed.toLocaleString();
  }

  // Update last update time
  const lastUpdate = document.getElementById("lastUpdate");
  if (lastUpdate) {
    lastUpdate.textContent = new Date().toLocaleTimeString();
  }

  // Update stats grid
  updateStatsGrid();
}

function updateComplianceUI(data) {
  // Update badge
  const status = data.is_compliant ? "COMPLIANT" : "NON-COMPLIANT";
  const badgeClass = data.is_compliant ? "compliant" : "non-compliant";
  const icon = data.is_compliant ? "âœ…" : "âš ï¸";

  updateComplianceBadge(status, badgeClass, icon);

  // Update current status
  const currentStatus = document.getElementById("currentStatus");
  if (currentStatus) {
    currentStatus.textContent = status;
    currentStatus.style.color = data.is_compliant ? "#22c55e" : "#ef4444";
  }

  // Update violations
  if (!data.is_compliant && data.violations && data.violations.length > 0) {
    showViolations(data.violations);
  } else {
    hideViolations();
  }

  // Update detected classes
  if (data.detected_classes && data.detected_classes.length > 0) {
    updateDetectedClasses(data.detected_classes);
  }

  // Update confidence scores
  if (data.confidence_scores) {
    updateConfidenceScores(data.confidence_scores);
  }
}

function updateComplianceBadge(text, className, icon) {
  const badge = document.getElementById("complianceBadge");
  const badgeText = document.getElementById("badgeText");
  const badgeIcon = document.getElementById("badgeIcon");

  if (badge && badgeText && badgeIcon) {
    badge.className = `compliance-badge ${className}`;
    badge.style.display = "inline-flex";
    badgeText.textContent = text;
    badgeIcon.textContent = icon;
  }
}

function updateStreamInfo(text, type) {
  const streamInfo = document.getElementById("streamInfo");
  if (streamInfo) {
    streamInfo.textContent = text;
    streamInfo.style.color =
      type === "success" ? "#22c55e" : type === "error" ? "#ef4444" : "#9ca3af";
  }
}

function showViolations(violations) {
  const card = document.getElementById("violationsCard");
  const list = document.getElementById("violationsList");

  if (card && list) {
    card.style.display = "block";
    list.innerHTML = violations.map((v) => `<li>${v}</li>`).join("");
  }
}

function hideViolations() {
  const card = document.getElementById("violationsCard");
  if (card) {
    card.style.display = "none";
  }
}

function updateDetectedClasses(classes) {
  const container = document.getElementById("detectedClasses");
  if (container) {
    container.innerHTML = classes
      .map((cls) => `<span class="class-badge">${cls}</span>`)
      .join("");
  }
}

function updateConfidenceScores(scores) {
  const container = document.getElementById("confidenceScores");
  if (container) {
    const html = Object.entries(scores)
      .map(
        ([cls, conf]) => `
                <div style="margin-bottom: 6px;">
                    <span style="color: #e5e7eb;">${cls}:</span>
                    <span style="color: #3b82f6; font-weight: 600;">${(
                      conf * 100
                    ).toFixed(1)}%</span>
                </div>
            `,
      )
      .join("");

    container.innerHTML =
      html || '<span style="color: #6b7280;">No detections</span>';
  }
}

function updateStatsGrid() {
  const updateElement = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value.toLocaleString();
  };

  updateElement("totalFrames", stats.totalFrames);
  updateElement("framesWithPerson", stats.framesWithPerson);
  updateElement("compliantFrames", stats.compliantFrames);
  updateElement("nonCompliantFrames", stats.nonCompliantFrames);
}

// ===============================
// Actions
// ===============================
function refreshStream() {
  const videoStream = document.getElementById("videoStream");
  if (videoStream) {
    // Force reload by adding timestamp
    const src = videoStream.src.split("?")[0];
    videoStream.src = `${src}?t=${Date.now()}`;

    showToast("ðŸ”„ Stream refreshed", 1500);
  }
}

// ===============================
// Toast Notifications
// ===============================
function showToast(message, timeoutMs = 2000) {
  const toast = document.getElementById("uploadToast");
  if (toast) {
    toast.textContent = message;
    toast.classList.remove("hidden");
    setTimeout(() => toast.classList.add("hidden"), timeoutMs);
  }
}

// ===============================
// Cleanup
// ===============================
window.addEventListener("beforeunload", () => {
  if (statusTimer) clearInterval(statusTimer);
  if (complianceTimer) clearInterval(complianceTimer);
});
