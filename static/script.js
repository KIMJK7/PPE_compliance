// ----- CONFIG -----
const CLASS_NAMES = [
  "exposed_hair",
  "exposed_ear",
  "head_net",
  "beard_net",
  "exposed_beard",
];

const MODEL_PATH =
  "https://huggingface.co/jk24122003/yolo_comp_non-comp/resolve/main/best.onnx";
const VIDEO_INTERVAL_MS = 300;
const LIVE_GALLERY_MAX = 40;
const LIVE_FEED_URL = "http://172.17.12.41:8080/video_feed";
const LIVE_INFER_INTERVAL_MS = 250;
const LIVE_CANVAS_W = 640;
const LIVE_CANVAS_H = 360;
const livePrepCanvas = document.createElement("canvas");
const livePrepCtx = livePrepCanvas.getContext("2d", {
  willReadFrequently: true,
});

// ----- BEEP ALERT SYSTEM -----
let beepEnabled = true;
let lastBeepTime = 0;
const BEEP_COOLDOWN_MS = 1000;

function playBeep(type = "warning") {
  if (!beepEnabled) return;

  const now = Date.now();
  if (now - lastBeepTime < BEEP_COOLDOWN_MS) return;
  lastBeepTime = now;

  try {
    const audioContext = new (
      window.AudioContext || window.webkitAudioContext
    )();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    if (type === "critical") {
      oscillator.frequency.value = 1000;
      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(
        0.01,
        audioContext.currentTime + 0.1,
      );
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.1);

      setTimeout(() => {
        const osc2 = audioContext.createOscillator();
        const gain2 = audioContext.createGain();
        osc2.connect(gain2);
        gain2.connect(audioContext.destination);
        osc2.frequency.value = 1000;
        gain2.gain.setValueAtTime(0.3, audioContext.currentTime);
        gain2.gain.exponentialRampToValueAtTime(
          0.01,
          audioContext.currentTime + 0.1,
        );
        osc2.start(audioContext.currentTime);
        osc2.stop(audioContext.currentTime + 0.1);
      }, 150);
    } else {
      oscillator.frequency.value = 800;
      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(
        0.01,
        audioContext.currentTime + 0.2,
      );
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.2);
    }
  } catch (error) {
    console.error("Error playing beep:", error);
  }
}

function toggleBeep() {
  beepEnabled = !beepEnabled;
  const status = beepEnabled ? "enabled" : "disabled";
  showToast(`ðŸ”” Alert sound ${status}`, 1500);

  const toggleBtn = document.getElementById("beepToggle");
  if (toggleBtn) {
    toggleBtn.textContent = beepEnabled ? "ðŸ”” Sound: ON" : "ðŸ”• Sound: OFF";
    toggleBtn.style.background = beepEnabled ? "#10b981" : "#6b7280";
  }
}

// ----- DASHBOARD INTEGRATION -----
async function sendFrameToDashboard(
  status,
  reason,
  classes,
  canvas,
  detections,
) {
  try {
    const avgConfidence =
      detections.length > 0
        ? detections.reduce((sum, det) => sum + det.conf, 0) / detections.length
        : 0;

    const imageData = canvas.toDataURL("image/png");

    const payload = {
      type: status === "COMPLIANT" ? "compliant" : "non_compliant",
      reason: reason,
      classes: classes,
      image_data: imageData,
      avg_confidence: avgConfidence,
    };

    const response = await fetch("/api/frames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      console.log("âœ“ Frame sent to dashboard:", status);
    }
  } catch (error) {
    console.error("Error sending frame to dashboard:", error);
  }
}

// ----- GLOBAL STATE -----
let liveRenderRaf = null;
let liveInferTimer = null;
let liveStreamPromise = null;
let session = null;
let modelInputName = null;
let confThreshold = 0.25;
let liveAbort = null;
let liveLatestBitmap = null;
let mode = null;
let imageElement = null;
let videoElement = null;
let mainCanvas = null;
let fileObjectUrl = null;
let videoProcessing = false;
let videoTimerId = null;
let videoFramesProcessed = 0;
let videoFramesCompliant = 0;
let videoFramesNonCompliant = 0;
let currentDetections = [];
let modelLoadProgress = 0;
let modelLoadTimer = null;
let liveImg = null;
let liveProcessing = false;
const LIVE_INPUT_SIZE = 640;
const LIVE_FRAME_SKIP = 1;
let liveFrameCounter = 0;
let liveInferenceBusy = false;
let lastLiveDetections = [];
let lastLiveStatus = null;
let lastLiveReason = "";
let videoInferenceBusy = false;
let liveStream = null;

// ----- UTILITY FUNCTIONS -----
function concatU8(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function findMarker(buf, marker, from = 0) {
  for (let i = from; i <= buf.length - marker.length; i++) {
    let ok = true;
    for (let j = 0; j < marker.length; j++) {
      if (buf[i + j] !== marker[j]) {
        ok = false;
        break;
      }
    }
    if (ok) return i;
  }
  return -1;
}

function getSourceDims(source) {
  if (source instanceof HTMLVideoElement) {
    return { w: source.videoWidth, h: source.videoHeight };
  }
  if (source instanceof HTMLImageElement) {
    return { w: source.naturalWidth, h: source.naturalHeight };
  }
  return { w: source.width, h: source.height };
}

function scaleDetectionsToLiveCanvas(detections, source) {
  const { w: sw, h: sh } = getSourceDims(source);
  const sx = LIVE_CANVAS_W / sw;
  const sy = LIVE_CANVAS_H / sh;
  return detections.map((det) => ({
    ...det,
    xyxy: [
      det.xyxy[0] * sx,
      det.xyxy[1] * sy,
      det.xyxy[2] * sx,
      det.xyxy[3] * sy,
    ],
  }));
}

// ----- COMPLIANCE LOGIC -----
function decideCompliance(detectedClassesSet) {
  const s = detectedClassesSet;
  if (s.has("exposed_hair")) {
    return { status: "NON-COMPLIANT", reason: "exposed_hair detected" };
  }
  if (s.has("exposed_beard")) {
    return { status: "NON-COMPLIANT", reason: "exposed_beard detected" };
  }
  if (s.has("head_net") && s.has("beard_net")) {
    return { status: "COMPLIANT", reason: "head_net and beard_net detected" };
  }
  if (s.has("exposed_ear")) {
    if (s.has("head_net")) {
      return {
        status: "COMPLIANT",
        reason: "exposed_ear present but head_net detected",
      };
    } else {
      return {
        status: "NON-COMPLIANT",
        reason: "exposed_ear with exposed_hair",
      };
    }
  }
  if (s.has("head_net")) {
    return { status: "COMPLIANT", reason: "head_net detected" };
  }
  return { status: "NON-COMPLIANT", reason: "required PPE not detected" };
}

// ----- UI HELPERS -----
function showToast(message, timeoutMs = 2000) {
  const toast = document.getElementById("uploadToast");
  if (toast) {
    toast.textContent = message;
    toast.classList.remove("hidden");
    setTimeout(() => toast.classList.add("hidden"), timeoutMs);
  }
}

function setFileLoading(isLoading, message = "Loading file...") {
  const loadingEl = document.getElementById("fileLoading");
  const fileInput = document.getElementById("fileInput");
  const runBtn = document.getElementById("runCheckBtn");
  const stopBtn = document.getElementById("stopBtn");

  if (isLoading) {
    loadingEl.textContent = message;
    loadingEl.classList.remove("hidden");
    fileInput.disabled = true;
    runBtn.disabled = true;
    stopBtn.disabled = true;
  } else {
    loadingEl.classList.add("hidden");
    fileInput.disabled = false;
  }
}

function clearFrameGallery() {
  const gallery = document.getElementById("frameGallery");
  if (gallery) gallery.innerHTML = "";
}

function openFrameModal(src) {
  const modal = document.getElementById("frameModal");
  const img = document.getElementById("frameModalImg");
  if (!modal || !img) return;
  img.src = src;
  modal.classList.remove("hidden");
}

function setupFrameModal() {
  const modal = document.getElementById("frameModal");
  if (!modal) return;
  modal.addEventListener("click", () => {
    modal.classList.add("hidden");
  });
}

function addFrameThumbnailFromCanvas(sourceCanvas, maxCount = null) {
  const gallery = document.getElementById("frameGallery");
  if (!gallery || !sourceCanvas) return;

  const fullCanvas = document.createElement("canvas");
  fullCanvas.width = sourceCanvas.width;
  fullCanvas.height = sourceCanvas.height;
  const fctx = fullCanvas.getContext("2d");
  fctx.drawImage(sourceCanvas, 0, 0);
  const fullSrc = fullCanvas.toDataURL("image/png");

  const img = document.createElement("img");
  img.src = fullSrc;
  img.className = "frame-thumb";
  img.addEventListener("click", () => openFrameModal(fullSrc));
  gallery.appendChild(img);

  if (maxCount && gallery.children.length > maxCount) {
    while (gallery.children.length > maxCount) {
      gallery.removeChild(gallery.firstChild);
    }
  }
}

function setStatusBorder(status) {
  const box = document.getElementById("statusBox");
  if (status === "COMPLIANT") {
    box.style.borderColor = "#16a34a";
  } else if (status === "NON-COMPLIANT") {
    box.style.borderColor = "#dc2626";
  } else {
    box.style.borderColor = "#1f2937";
  }
}

function updateDetectionsTable(detections) {
  const tbody = document.getElementById("detTableBody");
  tbody.innerHTML = "";
  detections.forEach((det, idx) => {
    const row = document.createElement("tr");
    const idxTd = document.createElement("td");
    idxTd.textContent = String(idx + 1);
    const clsTd = document.createElement("td");
    clsTd.textContent = det.class_name;
    const confTd = document.createElement("td");
    confTd.textContent = det.conf.toFixed(2);
    const boxTd = document.createElement("td");
    const [x1, y1, x2, y2] = det.xyxy.map((v) => v.toFixed(1));
    boxTd.textContent = `[${x1}, ${y1}, ${x2}, ${y2}]`;
    row.appendChild(idxTd);
    row.appendChild(clsTd);
    row.appendChild(confTd);
    row.appendChild(boxTd);
    tbody.appendChild(row);
  });
}

function resetStatusBox() {
  document.getElementById("statusModeText").textContent = "No file uploaded.";
  document.getElementById("statusMainText").textContent = "";
  document.getElementById("reasonText").textContent = "";
  document.getElementById("classesText").textContent = "-";
  document.getElementById("countText").textContent = "0";
  document.getElementById("framesProcessedText").textContent = "0";
  document.getElementById("framesCompliantText").textContent = "0";
  document.getElementById("framesNonCompliantText").textContent = "0";
  document.getElementById("progressText").textContent = "0%";
  setStatusBorder(null);
  updateDetectionsTable([]);
}

function setProgress(percent) {
  const p = Math.max(0, Math.min(100, percent || 0));
  document.getElementById("progressText").textContent = `${p.toFixed(1)}%`;
}

function clearCanvas() {
  const ctx = mainCanvas.getContext("2d");
  mainCanvas.width = 640;
  mainCanvas.height = 360;
  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
}

function drawSourceToCanvas(source) {
  const ctx = mainCanvas.getContext("2d");
  const { w: iw, h: ih } = getSourceDims(source);
  if (!iw || !ih) return;

  if (mode === "live") {
    if (
      mainCanvas.width !== LIVE_CANVAS_W ||
      mainCanvas.height !== LIVE_CANVAS_H
    ) {
      mainCanvas.width = LIVE_CANVAS_W;
      mainCanvas.height = LIVE_CANVAS_H;
    }
    ctx.drawImage(source, 0, 0, LIVE_CANVAS_W, LIVE_CANVAS_H);
    return;
  }

  mainCanvas.width = iw;
  mainCanvas.height = ih;
  ctx.drawImage(source, 0, 0, iw, ih);
}

function updateImageStatus(statusText, reason, classes, detections, status) {
  document.getElementById("statusModeText").textContent = "Mode: Image";
  document.getElementById("statusMainText").textContent = statusText || "";
  document.getElementById("reasonText").textContent = reason || "";
  document.getElementById("classesText").textContent = classes.length
    ? classes.join(", ")
    : "-";
  document.getElementById("countText").textContent =
    detections.length.toString();

  const processed = status === "COMPLIANT" || status === "NON-COMPLIANT";
  document.getElementById("framesProcessedText").textContent = processed
    ? "1"
    : "0";
  document.getElementById("framesCompliantText").textContent =
    processed && status === "COMPLIANT" ? "1" : "0";
  document.getElementById("framesNonCompliantText").textContent =
    processed && status === "NON-COMPLIANT" ? "1" : "0";

  updateDetectionsTable(detections);
  setStatusBorder(status);
}

function updateVideoStatus(
  frameStatusText,
  reason,
  framesProcessed,
  compliant,
  nonCompliant,
  status,
  progressPercent,
  detections,
) {
  const modeLabel = mode === "live" ? "Mode: Live Camera" : "Mode: Video";
  document.getElementById("statusModeText").textContent = modeLabel;
  document.getElementById("statusMainText").textContent = frameStatusText || "";
  document.getElementById("reasonText").textContent = reason || "";
  document.getElementById("classesText").textContent = "-";
  document.getElementById("countText").textContent =
    detections.length.toString();
  document.getElementById("framesProcessedText").textContent =
    framesProcessed.toString();
  document.getElementById("framesCompliantText").textContent =
    compliant.toString();
  document.getElementById("framesNonCompliantText").textContent =
    nonCompliant.toString();
  setProgress(progressPercent || 0);
  updateDetectionsTable(detections);
  setStatusBorder(status);
}

// ----- DRAWING FUNCTIONS -----
function drawDetections(source, detections, status, reason) {
  const ctx = mainCanvas.getContext("2d");
  const { w: iw, h: ih } = getSourceDims(source);
  if (!iw || !ih) return;
  mainCanvas.width = iw;
  mainCanvas.height = ih;
  ctx.drawImage(source, 0, 0, iw, ih);

  const bannerHeight = 50;
  const color = status === "COMPLIANT" ? "#16a34a" : "#dc2626";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, mainCanvas.width, bannerHeight);
  ctx.fillStyle = "white";
  ctx.font = "bold 20px system-ui";
  const text = status ? `${status} - ${reason}` : reason || "";
  ctx.fillText(text, 10, 32);

  ctx.lineWidth = 2;
  ctx.font = "14px system-ui";
  detections.forEach((det) => {
    const [x1, y1, x2, y2] = det.xyxy;
    const w = x2 - x1;
    const h = y2 - y1;
    ctx.strokeStyle = "#22c55e";
    ctx.strokeRect(x1, y1, w, h);
    const label = `${det.class_name} ${det.conf.toFixed(2)}`;
    ctx.fillStyle = "#22c55e";
    const textWidth = ctx.measureText(label).width;
    const textHeight = 16;
    const lx = x1;
    const ly = Math.max(0, y1 - textHeight - 4);
    ctx.fillRect(lx, ly, textWidth + 8, textHeight + 4);
    ctx.fillStyle = "black";
    ctx.fillText(label, lx + 4, ly + textHeight);
  });
}

function drawOverlayOnCanvas(status, reason, detections) {
  if (!mainCanvas) return;
  const ctx = mainCanvas.getContext("2d");
  const bannerHeight = 50;
  const color = status === "COMPLIANT" ? "#16a34a" : "#dc2626";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, mainCanvas.width, bannerHeight);
  ctx.fillStyle = "white";
  ctx.font = "bold 20px system-ui";
  const text = status ? `${status} - ${reason}` : reason || "";
  ctx.fillText(text, 10, 32);

  ctx.lineWidth = 2;
  ctx.font = "14px system-ui";
  detections.forEach((det) => {
    const [x1, y1, x2, y2] = det.xyxy;
    const w = x2 - x1;
    const h = y2 - y1;
    ctx.strokeStyle = "#22c55e";
    ctx.strokeRect(x1, y1, w, h);
    const label = `${det.class_name} ${det.conf.toFixed(2)}`;
    const textWidth = ctx.measureText(label).width;
    const textHeight = 16;
    const lx = x1;
    const ly = Math.max(0, y1 - textHeight - 4);
    ctx.fillStyle = "#22c55e";
    ctx.fillRect(lx, ly, textWidth + 8, textHeight + 4);
    ctx.fillStyle = "black";
    ctx.fillText(label, lx + 4, ly + textHeight);
  });
}

// ----- MODEL LOADER -----
function startModelLoader() {
  const overlay = document.getElementById("loaderOverlay");
  const bar = document.getElementById("loaderProgressBar");
  const label = document.getElementById("loaderProgressLabel");
  modelLoadProgress = 0;
  bar.style.width = "0%";
  label.textContent = "0%";
  overlay.style.display = "flex";

  if (modelLoadTimer) clearInterval(modelLoadTimer);
  modelLoadTimer = setInterval(() => {
    if (modelLoadProgress < 90) {
      modelLoadProgress += Math.random() * 5;
      if (modelLoadProgress > 90) modelLoadProgress = 90;
      bar.style.width = modelLoadProgress.toFixed(0) + "%";
      label.textContent = modelLoadProgress.toFixed(0) + "%";
    }
  }, 200);
}

function finishModelLoader(success) {
  const overlay = document.getElementById("loaderOverlay");
  const bar = document.getElementById("loaderProgressBar");
  const label = document.getElementById("loaderProgressLabel");

  if (modelLoadTimer) {
    clearInterval(modelLoadTimer);
    modelLoadTimer = null;
  }

  if (success) {
    modelLoadProgress = 100;
    bar.style.width = "100%";
    label.textContent = "100%";
    setTimeout(() => {
      overlay.style.opacity = "0";
      setTimeout(() => {
        overlay.style.display = "none";
      }, 300);
    }, 300);
  } else {
    label.textContent = "Error loading model";
  }
}

async function initModel() {
  startModelLoader();
  try {
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["webgpu", "webgl", "wasm"],
      graphOptimizationLevel: "all",
    });
    modelInputName = session.inputNames[0];
    console.log("Model loaded:", MODEL_PATH, "input:", modelInputName);

    const runBtn = document.getElementById("runCheckBtn");
    if (runBtn) runBtn.disabled = false;
    const fileInput = document.getElementById("fileInput");
    if (fileInput) fileInput.disabled = false;
    finishModelLoader(true);
  } catch (err) {
    console.error("Error loading model:", err);
    alert("Failed to load model. Please refresh the page.");
    finishModelLoader(false);
  }
}

// ----- PREPROCESSING -----
function preprocess(source, modelInputSize = 640) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const { w: iw, h: ih } = getSourceDims(source);
  if (!iw || !ih) {
    throw new Error("Source has no valid dimensions for preprocess.");
  }

  const scale = Math.min(modelInputSize / iw, modelInputSize / ih);
  const nw = Math.round(iw * scale);
  const nh = Math.round(ih * scale);
  canvas.width = modelInputSize;
  canvas.height = modelInputSize;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, modelInputSize, modelInputSize);
  const dx = Math.floor((modelInputSize - nw) / 2);
  const dy = Math.floor((modelInputSize - nh) / 2);
  ctx.drawImage(source, 0, 0, iw, ih, dx, dy, nw, nh);

  const imageData = ctx.getImageData(0, 0, modelInputSize, modelInputSize);
  const { data } = imageData;
  const floatData = new Float32Array(3 * modelInputSize * modelInputSize);
  for (let i = 0; i < modelInputSize * modelInputSize; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    floatData[i] = r;
    floatData[i + modelInputSize * modelInputSize] = g;
    floatData[i + 2 * modelInputSize * modelInputSize] = b;
  }

  const tensor = new ort.Tensor("float32", floatData, [
    1,
    3,
    modelInputSize,
    modelInputSize,
  ]);
  return {
    tensor,
    modelSize: modelInputSize,
    scale,
    dx,
    dy,
    origWidth: iw,
    origHeight: ih,
  };
}

function preprocessLive(source, modelInputSize = LIVE_INPUT_SIZE) {
  const { w: iw, h: ih } = getSourceDims(source);
  if (!iw || !ih) throw new Error("Live source has no dimensions");

  const scale = Math.min(modelInputSize / iw, modelInputSize / ih);
  const nw = Math.round(iw * scale);
  const nh = Math.round(ih * scale);
  livePrepCanvas.width = modelInputSize;
  livePrepCanvas.height = modelInputSize;
  livePrepCtx.fillStyle = "black";
  livePrepCtx.fillRect(0, 0, modelInputSize, modelInputSize);
  const dx = Math.floor((modelInputSize - nw) / 2);
  const dy = Math.floor((modelInputSize - nh) / 2);
  livePrepCtx.drawImage(source, 0, 0, iw, ih, dx, dy, nw, nh);

  const imageData = livePrepCtx.getImageData(
    0,
    0,
    modelInputSize,
    modelInputSize,
  );
  const { data } = imageData;
  const floatData = new Float32Array(3 * modelInputSize * modelInputSize);
  for (let i = 0; i < modelInputSize * modelInputSize; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    floatData[i] = r;
    floatData[i + modelInputSize * modelInputSize] = g;
    floatData[i + 2 * modelInputSize * modelInputSize] = b;
  }

  return {
    tensor: new ort.Tensor("float32", floatData, [
      1,
      3,
      modelInputSize,
      modelInputSize,
    ]),
    modelSize: modelInputSize,
    scale,
    dx,
    dy,
    origWidth: iw,
    origHeight: ih,
  };
}

function parseDetections(outputTensor, prep) {
  const data = outputTensor.data;
  const dims = outputTensor.dims;
  const numDetections = dims[1];
  const stride = 6;
  const detections = [];

  for (let i = 0; i < numDetections; i++) {
    const offset = i * stride;
    const x1 = data[offset + 0];
    const y1 = data[offset + 1];
    const x2 = data[offset + 2];
    const y2 = data[offset + 3];
    const score = data[offset + 4];
    const clsIdFloat = data[offset + 5];

    if (score < confThreshold) continue;

    const clsId = Math.round(clsIdFloat);
    const className = CLASS_NAMES[clsId] || `class_${clsId}`;
    const { scale, dx, dy, origWidth, origHeight } = prep;

    let boxX1 = (x1 - dx) / scale;
    let boxY1 = (y1 - dy) / scale;
    let boxX2 = (x2 - dx) / scale;
    let boxY2 = (y2 - dy) / scale;

    boxX1 = Math.max(0, Math.min(origWidth - 1, boxX1));
    boxY1 = Math.max(0, Math.min(origHeight - 1, boxY1));
    boxX2 = Math.max(0, Math.min(origWidth - 1, boxX2));
    boxY2 = Math.max(0, Math.min(origHeight - 1, boxY2));

    detections.push({
      xyxy: [boxX1, boxY1, boxX2, boxY2],
      conf: score,
      class_id: clsId,
      class_name: className,
    });
  }
  return detections;
}

// ----- FILE HANDLING -----
function handleFileChange(event) {
  const file = event.target.files[0];
  if (!file) return;

  stopVideoProcessing();
  handleStopLiveClick();

  if (fileObjectUrl) {
    URL.revokeObjectURL(fileObjectUrl);
    fileObjectUrl = null;
  }

  if (videoElement) {
    videoElement.pause();
    videoElement.currentTime = 0;
    videoElement.src = "";
    videoElement.load();
  }

  videoFramesProcessed = 0;
  videoFramesCompliant = 0;
  videoFramesNonCompliant = 0;
  videoProcessing = false;
  videoInferenceBusy = false;

  setProgress(0);
  updateDetectionsTable([]);
  clearFrameGallery();

  const runBtn = document.getElementById("runCheckBtn");
  const stopBtn = document.getElementById("stopBtn");
  runBtn.disabled = true;
  stopBtn.disabled = true;

  if (file.type.startsWith("image/")) {
    mode = "image";
    document.getElementById("statusModeText").textContent = "Mode: Image";
    loadImageFile(file);
  } else if (file.type.startsWith("video/")) {
    mode = "video";
    document.getElementById("statusModeText").textContent = "Mode: Video";
    loadVideoFile(file);
  } else {
    alert("Please select an image or video file.");
    mode = null;
  }
}

function loadImageFile(file) {
  const fileInput = document.getElementById("fileInput");
  fileInput.disabled = false;

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      imageElement = img;
      drawSourceToCanvas(img);
      document.getElementById("runCheckBtn").disabled = false;
      document.getElementById("stopBtn").disabled = true;
      updateImageStatus("Image loaded. Click 'Run Check'.", "", [], [], null);
      setProgress(0);
      showToast("Image uploaded successfully");
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function loadVideoFile(file) {
  console.log(
    "[video] Selected file:",
    file.name,
    file.type,
    file.size,
    "bytes",
  );

  setFileLoading(true, "Loading video into browser...");

  if (fileObjectUrl) {
    URL.revokeObjectURL(fileObjectUrl);
  }
  fileObjectUrl = URL.createObjectURL(file);
  videoElement.src = fileObjectUrl;
  videoElement.load();

  videoElement.onloadedmetadata = () => {
    console.log(
      "[video] onloadedmetadata:",
      videoElement.videoWidth,
      videoElement.videoHeight,
      "duration:",
      videoElement.duration,
    );
  };

  videoElement.onloadeddata = () => {
    console.log(
      "[video] onloadeddata:",
      videoElement.videoWidth,
      videoElement.videoHeight,
    );
    const vw = videoElement.videoWidth;
    const vh = videoElement.videoHeight;

    if (!vw || !vh) {
      setFileLoading(false);
      updateVideoStatus(
        "Failed to load video (no dimensions).",
        "",
        0,
        0,
        0,
        null,
        0,
        [],
      );
      return;
    }

    drawSourceToCanvas(videoElement);
    setFileLoading(false);
    document.getElementById("runCheckBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    updateVideoStatus(
      "Video loaded. Click 'Run Check'.",
      "",
      0,
      0,
      0,
      null,
      0,
      [],
    );
    showToast("Video uploaded successfully");
  };

  videoElement.onerror = () => {
    console.error("[video] Error loading video:", videoElement.error);
    setFileLoading(false);
    updateVideoStatus(
      "Error loading video. Try an .mp4 (H.264) file.",
      "",
      0,
      0,
      0,
      null,
      0,
      [],
    );
  };
}

// ----- IMAGE PROCESSING -----
async function handleRunClick() {
  if (!session || !mode) return;

  const runBtn = document.getElementById("runCheckBtn");
  const stopBtn = document.getElementById("stopBtn");

  if (mode === "image") {
    if (!imageElement) return;
    runBtn.disabled = true;
    stopBtn.disabled = true;

    try {
      const prep = preprocess(imageElement, 640);
      const feeds = {};
      feeds[modelInputName] = prep.tensor;

      const results = await session.run(feeds);
      const outputName = session.outputNames[0];
      const outputTensor = results[outputName];

      const detections = parseDetections(outputTensor, prep);
      currentDetections = detections;

      const classesSet = new Set(detections.map((d) => d.class_name));
      const { status, reason } = decideCompliance(classesSet);

      if (status === "NON-COMPLIANT") {
        playBeep("warning");
      }

      drawDetections(imageElement, detections, status, reason);
      updateImageStatus(
        `Status: ${status}`,
        reason,
        Array.from(classesSet),
        detections,
        status,
      );
      setProgress(100);

      await sendFrameToDashboard(
        status,
        reason,
        Array.from(classesSet),
        mainCanvas,
        detections,
      );
    } catch (err) {
      console.error("Error during image inference:", err);
      alert("Error during inference. Check console.");
    } finally {
      runBtn.disabled = false;
    }
  } else if (mode === "video") {
    startVideoProcessing();
  }
}

// ----- VIDEO PROCESSING -----
function startVideoProcessing() {
  if (!session || !videoElement || !videoElement.src) return;
  if (videoProcessing) return;

  videoFramesProcessed = 0;
  videoFramesCompliant = 0;
  videoFramesNonCompliant = 0;
  setProgress(0);
  videoProcessing = true;
  videoInferenceBusy = false;
  document.getElementById("runCheckBtn").disabled = true;
  document.getElementById("stopBtn").disabled = false;

  videoElement.currentTime = 0;
  videoElement
    .play()
    .then(() => {
      videoTimerId = setInterval(processVideoFrame, VIDEO_INTERVAL_MS);
    })
    .catch((err) => {
      console.error("Error playing video:", err);
      updateVideoStatus(
        "Error starting video playback.",
        String(err),
        0,
        0,
        0,
        null,
        0,
        [],
      );
      stopVideoProcessing();
    });
}

function stopVideoProcessing() {
  if (!videoProcessing) return;
  videoProcessing = false;
  videoInferenceBusy = false;

  if (videoTimerId) {
    clearInterval(videoTimerId);
    videoTimerId = null;
  }

  if (videoElement) {
    if (!videoElement.paused) {
      videoElement.pause();
    }
    videoElement.currentTime = 0;
  }

  document.getElementById("runCheckBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;
}

async function processVideoFrame() {
  if (!videoProcessing || !videoElement) return;
  if (videoInferenceBusy) return;
  videoInferenceBusy = true;

  try {
    if (videoElement.ended) {
      updateVideoStatus(
        "Video finished.",
        "",
        videoFramesProcessed,
        videoFramesCompliant,
        videoFramesNonCompliant,
        null,
        100,
        currentDetections,
      );
      stopVideoProcessing();

      const fileInput = document.getElementById("fileInput");
      if (fileInput) {
        fileInput.disabled = false;
      }

      showToast(
        "Video processing complete. You can now upload a new file.",
        3000,
      );
      return;
    }

    if (videoElement.readyState < 2) {
      return;
    }

    const fw = videoElement.videoWidth;
    const fh = videoElement.videoHeight;
    if (!fw || !fh) return;

    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = fw;
    frameCanvas.height = fh;
    const fctx = frameCanvas.getContext("2d");
    fctx.drawImage(videoElement, 0, 0, fw, fh);

    const prep = preprocess(frameCanvas, 640);
    const feeds = {};
    feeds[modelInputName] = prep.tensor;

    const results = await session.run(feeds);
    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];

    const detections = parseDetections(outputTensor, prep);
    currentDetections = detections;

    const classesSet = new Set(detections.map((d) => d.class_name));
    const { status, reason } = decideCompliance(classesSet);

    if (status === "NON-COMPLIANT") {
      playBeep("warning");
    }

    videoFramesProcessed++;
    if (status === "COMPLIANT") videoFramesCompliant++;
    else videoFramesNonCompliant++;

    drawDetections(frameCanvas, detections, status, reason);

    if (videoFramesProcessed % 5 === 0) {
      await sendFrameToDashboard(
        status,
        reason,
        Array.from(classesSet),
        mainCanvas,
        detections,
      );
    }

    addFrameThumbnailFromCanvas(mainCanvas);

    let progress = 0;
    if (videoElement.duration > 0) {
      progress = (videoElement.currentTime / videoElement.duration) * 100;
    }

    updateVideoStatus(
      `Current frame: ${status}`,
      reason,
      videoFramesProcessed,
      videoFramesCompliant,
      videoFramesNonCompliant,
      status,
      progress,
      detections,
    );
  } catch (err) {
    console.error("Error during video frame inference:", err);
    updateVideoStatus(
      "Error during video processing.",
      String(err),
      videoFramesProcessed,
      videoFramesCompliant,
      videoFramesNonCompliant,
      null,
      0,
      currentDetections,
    );
    stopVideoProcessing();
  } finally {
    videoInferenceBusy = false;
  }
}

// ----- LIVE FEED -----
async function startLiveFetchStream() {
  liveAbort = new AbortController();
  const res = await fetch(LIVE_FEED_URL, {
    signal: liveAbort.signal,
    cache: "no-store",
  });
  if (!res.ok || !res.body) {
    throw new Error("Live feed fetch failed");
  }

  const reader = res.body.getReader();
  let buffer = new Uint8Array(0);
  const SOI = new Uint8Array([0xff, 0xd8]);
  const EOI = new Uint8Array([0xff, 0xd9]);

  try {
    while (liveProcessing) {
      const { value, done } = await reader.read();
      if (done || !liveProcessing) break;
      if (!value) continue;

      buffer = concatU8(buffer, value);
      let start = findMarker(buffer, SOI, 0);
      let end = findMarker(buffer, EOI, start + 2);

      while (start !== -1 && end !== -1 && end > start) {
        const jpg = buffer.slice(start, end + 2);
        buffer = buffer.slice(end + 2);

        try {
          const blob = new Blob([jpg], { type: "image/jpeg" });
          const bitmap = await createImageBitmap(blob);
          if (!liveProcessing) {
            if (bitmap?.close) bitmap.close();
            break;
          }
          if (liveLatestBitmap?.close) liveLatestBitmap.close();
          liveLatestBitmap = bitmap;
        } catch {}

        start = findMarker(buffer, SOI, 0);
        end = findMarker(buffer, EOI, start + 2);
      }
    }
  } catch (e) {
    if (liveProcessing) console.error("Live stream error:", e);
  } finally {
    try {
      reader.cancel();
    } catch {}
  }
}

const LIVE_RENDER_INTERVAL_MS = 80;
let lastRenderTs = 0;

function startLiveRenderLoop() {
  const loop = (ts) => {
    if (!liveProcessing) return;

    if (ts - lastRenderTs >= LIVE_RENDER_INTERVAL_MS) {
      lastRenderTs = ts;
      if (liveLatestBitmap) {
        drawSourceToCanvas(liveLatestBitmap);
        if (lastLiveStatus) {
          const scaled = scaleDetectionsToLiveCanvas(
            lastLiveDetections,
            liveLatestBitmap,
          );
          drawOverlayOnCanvas(lastLiveStatus, lastLiveReason, scaled);
        }
      }
    }
    liveRenderRaf = requestAnimationFrame(loop);
  };
  liveRenderRaf = requestAnimationFrame(loop);
}

function startLiveInferenceTimer() {
  if (liveInferTimer) clearInterval(liveInferTimer);

  liveInferTimer = setInterval(async () => {
    if (!liveProcessing) return;
    if (!liveLatestBitmap) return;
    if (liveInferenceBusy) return;

    liveInferenceBusy = true;
    try {
      await runLiveInferenceFromSource(liveLatestBitmap);
    } catch (e) {
      console.error("Live inference error:", e);
    } finally {
      liveInferenceBusy = false;
    }
  }, LIVE_INFER_INTERVAL_MS);
}

const LIVE_UI_EVERY_N = 2;

async function runLiveInferenceFromSource(source) {
  if (!session || !source) return;

  const prep = preprocessLive(source, LIVE_INPUT_SIZE);
  const feeds = {};
  feeds[modelInputName] = prep.tensor;

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const outputTensor = results[outputName];

  const detections = parseDetections(outputTensor, prep);
  currentDetections = detections;
  lastLiveDetections = detections;

  const classesSet = new Set(detections.map((d) => d.class_name));
  const { status, reason } = decideCompliance(classesSet);

  lastLiveStatus = status;
  lastLiveReason = reason;

  if (status === "NON-COMPLIANT") {
    playBeep("warning");
  }

  videoFramesProcessed++;
  if (status === "COMPLIANT") videoFramesCompliant++;
  else videoFramesNonCompliant++;

  const shouldUpdateUI = videoFramesProcessed % LIVE_UI_EVERY_N === 0;
  if (shouldUpdateUI) {
    updateVideoStatus(
      `Live frame: ${status}`,
      reason,
      videoFramesProcessed,
      videoFramesCompliant,
      videoFramesNonCompliant,
      status,
      0,
      detections,
    );
  }

  if (videoFramesProcessed % 10 === 0) {
    await sendFrameToDashboard(
      status,
      reason,
      Array.from(classesSet),
      mainCanvas,
      detections,
    );
  }

  if (videoFramesProcessed % 3 === 0) {
    addFrameThumbnailFromCanvas(mainCanvas, LIVE_GALLERY_MAX);
  }
}

async function handleStartLiveClick() {
  stopVideoProcessing();

  videoFramesProcessed = 0;
  videoFramesCompliant = 0;
  videoFramesNonCompliant = 0;
  setProgress(0);
  updateDetectionsTable([]);
  clearFrameGallery();

  mode = "live";
  document.getElementById("statusModeText").textContent = "Mode: Live Feed";

  const startBtn = document.getElementById("startLiveBtn");
  const stopLiveBtn = document.getElementById("stopLiveBtn");
  const fileInput = document.getElementById("fileInput");
  const runBtn = document.getElementById("runCheckBtn");
  const stopBtn = document.getElementById("stopBtn");

  startBtn.disabled = true;
  stopLiveBtn.disabled = false;
  fileInput.disabled = true;
  runBtn.disabled = true;
  stopBtn.disabled = true;

  liveProcessing = true;
  liveFrameCounter = 0;
  liveInferenceBusy = false;
  lastLiveDetections = [];
  lastLiveStatus = null;
  lastLiveReason = "";

  updateVideoStatus("Live feed started.", "", 0, 0, 0, null, 0, []);

  startLiveRenderLoop();
  startLiveInferenceTimer();

  liveStreamPromise = startLiveFetchStream().catch((e) => {
    if (liveProcessing) {
      console.error(e);
      updateVideoStatus("Live feed failed.", String(e), 0, 0, 0, null, 0, []);
      handleStopLiveClick();
    }
  });
}

function handleStopLiveClick() {
  liveProcessing = false;

  if (liveAbort) {
    try {
      liveAbort.abort();
    } catch {}
    liveAbort = null;
  }

  if (liveRenderRaf) {
    cancelAnimationFrame(liveRenderRaf);
    liveRenderRaf = null;
  }

  if (liveInferTimer) {
    clearInterval(liveInferTimer);
    liveInferTimer = null;
  }

  if (liveLatestBitmap?.close) liveLatestBitmap.close();
  liveLatestBitmap = null;

  const startBtn = document.getElementById("startLiveBtn");
  const stopLiveBtn = document.getElementById("stopLiveBtn");
  const fileInput = document.getElementById("fileInput");
  const runBtn = document.getElementById("runCheckBtn");
  const stopBtn = document.getElementById("stopBtn");

  if (startBtn) startBtn.disabled = false;
  if (stopLiveBtn) stopLiveBtn.disabled = true;
  if (fileInput) fileInput.disabled = false;
  if (runBtn) runBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = true;

  mode = null;
}

// ----- CONTROL HANDLERS -----
function handleStopClick() {
  stopVideoProcessing();
}

function handleResetClick() {
  stopVideoProcessing();
  handleStopLiveClick();

  liveFrameCounter = 0;
  liveInferenceBusy = false;
  lastLiveDetections = [];
  lastLiveStatus = null;
  lastLiveReason = "";

  if (videoElement) {
    videoElement.pause();
    videoElement.currentTime = 0;
    videoElement.src = "";
    videoElement.load();
  }

  if (fileObjectUrl) {
    URL.revokeObjectURL(fileObjectUrl);
    fileObjectUrl = null;
  }

  const fileInput = document.getElementById("fileInput");
  const runBtn = document.getElementById("runCheckBtn");
  const stopBtn = document.getElementById("stopBtn");
  const startLiveBtn = document.getElementById("startLiveBtn");
  const stopLiveBtn = document.getElementById("stopLiveBtn");

  if (fileInput) {
    fileInput.value = "";
    fileInput.disabled = false;
  }
  if (runBtn) runBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = true;
  if (startLiveBtn) startLiveBtn.disabled = false;
  if (stopLiveBtn) stopLiveBtn.disabled = true;

  mode = null;
  imageElement = null;
  currentDetections = [];

  videoFramesProcessed = 0;
  videoFramesCompliant = 0;
  videoFramesNonCompliant = 0;
  videoProcessing = false;
  videoInferenceBusy = false;

  setProgress(0);
  clearCanvas();
  resetStatusBox();
  clearFrameGallery();

  console.log("âœ“ Reset complete - ready for new file");
}

// ----- SETUP -----
function setup() {
  mainCanvas = document.getElementById("mainCanvas");
  videoElement = document.getElementById("hiddenVideo");

  clearCanvas();
  resetStatusBox();

  const fileInput = document.getElementById("fileInput");
  const runBtn = document.getElementById("runCheckBtn");
  const stopBtn = document.getElementById("stopBtn");
  const resetBtn = document.getElementById("resetBtn");
  const slider = document.getElementById("confSlider");
  const confLabel = document.getElementById("confValue");
  const startLiveBtn = document.getElementById("startLiveBtn");
  const stopLiveBtn = document.getElementById("stopLiveBtn");

  if (startLiveBtn)
    startLiveBtn.addEventListener("click", handleStartLiveClick);
  if (stopLiveBtn) stopLiveBtn.addEventListener("click", handleStopLiveClick);
  fileInput.addEventListener("change", handleFileChange);
  runBtn.addEventListener("click", handleRunClick);
  stopBtn.addEventListener("click", handleStopClick);
  resetBtn.addEventListener("click", handleResetClick);

  confLabel.textContent = slider.value;
  confThreshold = parseFloat(slider.value);
  slider.addEventListener("input", () => {
    confLabel.textContent = slider.value;
    confThreshold = parseFloat(slider.value);
  });

  setupFrameModal();
  initModel().catch((err) => {
    console.error("Error loading model:", err);
    alert("Failed to load model. Check console.");
  });
}

document.addEventListener("DOMContentLoaded", setup);

/* -------------------------------
   Dashboard grouped by track_id
   (Top-K per track) renderer
-------------------------------- */
(function () {
  const nonGrid = document.getElementById("nonCompliantGrid");
  const comGrid = document.getElementById("compliantGrid");
  if (!nonGrid || !comGrid) return; // not on dashboard page

  const totalEl = document.getElementById("totalFrames");
  const compliantCountEl = document.getElementById("compliantCount");
  const nonCompliantCountEl = document.getElementById("nonCompliantCount");
  const rateEl = document.getElementById("complianceRate");
  const compliantBadgeEl = document.getElementById("compliantBadge");
  const nonCompliantBadgeEl = document.getElementById("nonCompliantBadge");
  const loadingEl = document.getElementById("loadingIndicator");

  function showLoading(show) {
    if (!loadingEl) return;
    if (show) loadingEl.classList.add("active");
    else loadingEl.classList.remove("active");
  }

  function escapeHtml(s) {
    return String(s || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function flattenTrackMap(trackMap) {
    const out = [];
    if (!trackMap) return out;
    Object.keys(trackMap).forEach((tid) => {
      const arr = trackMap[tid] || [];
      arr.forEach((fr) => out.push(fr));
    });
    return out;
  }

  function updateStats(data) {
    const compliantTracks = data.compliant_tracks || null;
    const nonTracks = data.non_compliant_tracks || null;

    const compliantFlat =
      (compliantTracks ? flattenTrackMap(compliantTracks) : null) ||
      data.compliant ||
      [];
    const nonFlat =
      (nonTracks ? flattenTrackMap(nonTracks) : null) ||
      data.non_compliant ||
      [];

    const total =
      typeof data.total === "number"
        ? data.total
        : compliantFlat.length + nonFlat.length;

    const compliant = compliantFlat.length;
    const nonCompliant = nonFlat.length;
    const rate = total > 0 ? ((compliant / total) * 100).toFixed(1) : "0.0";

    if (totalEl) totalEl.textContent = String(total);
    if (compliantCountEl) compliantCountEl.textContent = String(compliant);
    if (nonCompliantCountEl)
      nonCompliantCountEl.textContent = String(nonCompliant);
    if (rateEl) rateEl.textContent = rate + "%";
    if (compliantBadgeEl) compliantBadgeEl.textContent = String(compliant);
    if (nonCompliantBadgeEl)
      nonCompliantBadgeEl.textContent = String(nonCompliant);
  }

  function frameCardHtml(frame, idx, trackId, bucketLabel) {
    const img = frame.image_data || "";
    const ts = escapeHtml(frame.timestamp || "");
    const reason = escapeHtml(frame.reason || "");
    const classes = (frame.classes || []).map(escapeHtml).join(", ");
    const score = Number(frame.avg_confidence || 0);
    const scorePct = (score * 100).toFixed(1) + "%";
    const tid = escapeHtml(trackId || frame.track_id || "unknown");

    return `
      <div class="frame-card ${bucketLabel}">
        <img src="${img}" alt="Frame ${idx + 1}"
             class="frame-image"
             onclick="openModal('${img}')">
        <div class="frame-details">
          <div class="frame-title">
            ${bucketLabel === "non-compliant" ? "Non-Compliant" : "Compliant"}
            — ${tid} — Frame #${idx + 1}
          </div>
          <div class="frame-meta">
            <div><span class="label">Time:</span> ${ts}</div>
            <div><span class="label">Reason:</span> ${reason || "-"}</div>
            <div><span class="label">Classes:</span> ${classes || "-"}</div>
            <div><span class="label">Score:</span> ${scorePct}</div>
          </div>
        </div>
      </div>
    `;
  }

  function renderGrouped(gridEl, trackMap, legacyArr, bucketLabel) {
    if (trackMap && Object.keys(trackMap).length > 0) {
      const html = Object.keys(trackMap)
        .sort()
        .map((tid) => {
          const frames = trackMap[tid] || [];
          const cards = frames
            .map((fr, i) => frameCardHtml(fr, i, tid, bucketLabel))
            .join("");
          return `
            <div style="grid-column: 1 / -1; margin: 10px 0 4px;">
              <div style="font-weight: 700; color: #e5e7eb;">
                Person/Track: ${escapeHtml(tid)} (Top ${frames.length})
              </div>
            </div>
            ${cards}
          `;
        })
        .join("");
      gridEl.innerHTML = html;
      return;
    }

    const arr = legacyArr || [];
    if (arr.length === 0) {
      gridEl.innerHTML = `<div style="grid-column:1/-1; color:#9ca3af;">No frames captured.</div>`;
      return;
    }

    gridEl.innerHTML = arr
      .map((fr, i) =>
        frameCardHtml(fr, i, fr.track_id || "unknown", bucketLabel),
      )
      .join("");
  }

  async function refreshDashboardData() {
    showLoading(true);
    try {
      const res = await fetch("/api/frames", { cache: "no-store" });
      const data = await res.json();
      updateStats(data);

      renderGrouped(
        nonGrid,
        data.non_compliant_tracks,
        data.non_compliant,
        "non-compliant",
      );
      renderGrouped(
        comGrid,
        data.compliant_tracks,
        data.compliant,
        "compliant",
      );
    } catch (e) {
      console.error("Dashboard refresh failed:", e);
      alert("Failed to load frames");
    } finally {
      showLoading(false);
    }
  }

  // Expose the same name used by dashboard buttons
  window.refreshData = refreshDashboardData;

  document.addEventListener("DOMContentLoaded", refreshDashboardData);
})();
