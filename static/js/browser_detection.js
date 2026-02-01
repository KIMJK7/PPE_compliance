// ==================== CONFIG ====================
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
const GALLERY_MAX_ITEMS = 40;

// ==================== GLOBAL STATE ====================
let session = null;
let modelInputName = null;
let confThreshold = 0.25;
let mode = null; // 'image' or 'video'
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
let beepEnabled = true;
let lastBeepTime = 0;
const BEEP_COOLDOWN_MS = 1000;

// ==================== UTILITY FUNCTIONS ====================
function getSourceDims(source) {
  if (source instanceof HTMLVideoElement) {
    return { w: source.videoWidth, h: source.videoHeight };
  }
  if (source instanceof HTMLImageElement) {
    return { w: source.naturalWidth, h: source.naturalHeight };
  }
  return { w: source.width, h: source.height };
}

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
    if (fileInput) fileInput.disabled = true;
    if (runBtn) runBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
  } else {
    loadingEl.classList.add("hidden");
    if (fileInput) fileInput.disabled = false;
  }
}

// ==================== BEEP ALERT SYSTEM ====================
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
      // Double beep for critical
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
      // Single beep for warning
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
  showToast(`🔊 Alert sound ${status}`, 1500);

  const toggleBtn = document.getElementById("beepToggle");
  if (toggleBtn) {
    toggleBtn.textContent = beepEnabled ? "🔊 Sound: ON" : "🔕 Sound: OFF";
    toggleBtn.style.background = beepEnabled ? "#10b981" : "#6b7280";
  }
}

// ==================== DASHBOARD INTEGRATION ====================
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
      console.log("✓ Frame sent to dashboard:", status);
    }
  } catch (error) {
    console.error("Error sending frame to dashboard:", error);
  }
}

// ==================== COMPLIANCE LOGIC ====================
function decideCompliance(detectedClassesSet) {
  const s = detectedClassesSet;

  // ✅ NEW: If nothing from your 5 classes was detected
  if (!s || s.size === 0) {
    return { status: "NO-PERSON", reason: "person not detected" };
  }

  // Priority 1: Check for exposed violations
  if (s.has("exposed_hair")) {
    return { status: "NON-COMPLIANT", reason: "exposed_hair detected" };
  }
  if (s.has("exposed_beard")) {
    return { status: "NON-COMPLIANT", reason: "exposed_beard detected" };
  }

  // Priority 2: Check for full compliance
  if (s.has("head_net") && s.has("beard_net")) {
    return { status: "COMPLIANT", reason: "head_net and beard_net detected" };
  }

  // Priority 3: Handle exposed_ear cases
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

  // Priority 4: Check for head_net only
  if (s.has("head_net")) {
    return { status: "COMPLIANT", reason: "head_net detected" };
  }

  // Default: Non-compliant
  return { status: "NON-COMPLIANT", reason: "required PPE not detected" };
}

// ==================== MODEL LOADING ====================
async function initModel() {
  const overlay = document.getElementById("loaderOverlay");
  const progressBar = document.getElementById("loaderProgressBar");
  const progressLabel = document.getElementById("loaderProgressLabel");

  try {
    console.log("Loading YOLO model from:", MODEL_PATH);

    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 5;
      if (progress <= 90) {
        progressBar.style.width = progress + "%";
        progressLabel.textContent = progress + "%";
      }
    }, 100);

    // Load model
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });

    clearInterval(progressInterval);
    progressBar.style.width = "100%";
    progressLabel.textContent = "100%";

    modelInputName = session.inputNames[0];
    console.log("✓ Model loaded successfully");
    console.log("Input name:", modelInputName);
    console.log("Input shape:", session.inputMetadata);

    // Hide overlay after short delay
    setTimeout(() => {
      overlay.classList.add("hidden");
    }, 500);
  } catch (error) {
    console.error("Model loading failed:", error);
    alert("Failed to load model. Check console for details.");
  }
}

// ==================== PREPROCESSING ====================
function preprocess(source, inputSize = 640) {
  const { w, h } = getSourceDims(source);

  // Create temporary canvas for preprocessing
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = inputSize;
  tempCanvas.height = inputSize;
  const ctx = tempCanvas.getContext("2d", { willReadFrequently: true });

  // Letterbox: maintain aspect ratio
  const scale = Math.min(inputSize / w, inputSize / h);
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);
  const x = Math.floor((inputSize - nw) / 2);
  const y = Math.floor((inputSize - nh) / 2);

  // Fill black background
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, inputSize, inputSize);

  // Draw scaled image
  ctx.drawImage(source, x, y, nw, nh);

  // Get pixel data
  const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
  const pixels = imageData.data;

  // Convert to float32 CHW format [1, 3, 640, 640]
  const float32Data = new Float32Array(3 * inputSize * inputSize);
  for (let i = 0; i < inputSize * inputSize; i++) {
    float32Data[i] = pixels[i * 4] / 255.0; // R
    float32Data[inputSize * inputSize + i] = pixels[i * 4 + 1] / 255.0; // G
    float32Data[2 * inputSize * inputSize + i] = pixels[i * 4 + 2] / 255.0; // B
  }

  const tensor = new ort.Tensor("float32", float32Data, [
    1,
    3,
    inputSize,
    inputSize,
  ]);

  return { tensor, scale, x, y, nw, nh, origWidth: w, origHeight: h };
}

// ==================== DETECTION PARSING ====================
function parseDetections(outputTensor, prepData) {
  const data = outputTensor.data;
  const dims = outputTensor.dims;

  if (!dims || dims.length !== 3) {
    console.error("Unexpected output dims:", dims);
    return [];
  }

  const { scale, x, y, origWidth, origHeight } = prepData;

  // ---------- Case A: already NMSed output: [1, N, 6] ----------
  if (dims[2] === 6) {
    const N = dims[1];
    const dets = [];

    for (let i = 0; i < N; i++) {
      const off = i * 6;

      const x1m = data[off + 0];
      const y1m = data[off + 1];
      const x2m = data[off + 2];
      const y2m = data[off + 3];
      const score = data[off + 4];
      const clsId = Math.round(data[off + 5]);

      if (score < confThreshold) continue;

      // map from letterboxed model coords -> original coords
      let x1 = (x1m - x) / scale;
      let y1 = (y1m - y) / scale;
      let x2 = (x2m - x) / scale;
      let y2 = (y2m - y) / scale;

      // clamp
      x1 = Math.max(0, Math.min(origWidth - 1, x1));
      y1 = Math.max(0, Math.min(origHeight - 1, y1));
      x2 = Math.max(0, Math.min(origWidth - 1, x2));
      y2 = Math.max(0, Math.min(origHeight - 1, y2));

      dets.push({
        class_id: clsId,
        class_name: CLASS_NAMES[clsId] || `class_${clsId}`,
        conf: score,
        xyxy: [x1, y1, x2, y2],
      });
    }

    // already NMSed, but you may keep a light NMS
    return applyNMS(dets, 0.45);
  }

  // ---------- Case B: raw anchor format with 8400 ----------
  const has8400 = dims[1] === 8400 || dims[2] === 8400;
  if (!has8400) {
    console.error("Unexpected output shape:", dims);
    return [];
  }

  // Layout:
  // [1, F, 8400] => transpose=true
  // [1, 8400, F] => transpose=false
  const transpose = dims[2] === 8400;
  const numDetections = 8400;
  const F = transpose ? dims[1] : dims[2];

  const nc = CLASS_NAMES.length; // 5 classes
  const isV8 = F === 4 + nc; // 9 fields (cx,cy,w,h + 5 classes)
  const isV5 = F === 5 + nc; // 10 fields (cx,cy,w,h,obj + 5 classes)

  if (!isV8 && !isV5) {
    console.error(
      `Field mismatch. Got F=${F} but expected 4+nc=${4 + nc} or 5+nc=${5 + nc}. dims=`,
      dims,
    );
    return [];
  }

  const read = (i, fieldIdx) => {
    // i = anchor index (0..8399)
    return transpose
      ? data[fieldIdx * numDetections + i] // [F,8400]
      : data[i * F + fieldIdx]; // [8400,F]
  };

  const detections = [];

  for (let i = 0; i < numDetections; i++) {
    const cx = read(i, 0);
    const cy = read(i, 1);
    const w = read(i, 2);
    const h = read(i, 3);

    let obj = 1.0;
    let classStart = 4;

    if (isV5) {
      obj = read(i, 4);
      classStart = 5;
    }

    // pick best class
    let bestScore = -Infinity;
    let bestCls = -1;
    for (let c = 0; c < nc; c++) {
      const sc = read(i, classStart + c);
      if (sc > bestScore) {
        bestScore = sc;
        bestCls = c;
      }
    }

    const conf = obj * bestScore;
    if (conf < confThreshold) continue;

    // raw -> xyxy in model (letterbox) coords
    const x1m = cx - w / 2;
    const y1m = cy - h / 2;
    const x2m = cx + w / 2;
    const y2m = cy + h / 2;

    // map -> original coords
    let x1 = (x1m - x) / scale;
    let y1 = (y1m - y) / scale;
    let x2 = (x2m - x) / scale;
    let y2 = (y2m - y) / scale;

    // clamp
    x1 = Math.max(0, Math.min(origWidth - 1, x1));
    y1 = Math.max(0, Math.min(origHeight - 1, y1));
    x2 = Math.max(0, Math.min(origWidth - 1, x2));
    y2 = Math.max(0, Math.min(origHeight - 1, y2));

    detections.push({
      class_id: bestCls,
      class_name: CLASS_NAMES[bestCls] || `class_${bestCls}`,
      conf,
      xyxy: [x1, y1, x2, y2],
    });
  }

  // NMS to remove duplicates
  return applyNMS(detections, 0.45);
}

function applyNMS(detections, iouThreshold = 0.45) {
  if (detections.length === 0) return [];

  // Sort by confidence
  detections.sort((a, b) => b.conf - a.conf);

  const keep = [];
  const suppressed = new Set();

  for (let i = 0; i < detections.length; i++) {
    if (suppressed.has(i)) continue;
    keep.push(detections[i]);

    const boxA = detections[i].xyxy;
    for (let j = i + 1; j < detections.length; j++) {
      if (suppressed.has(j)) continue;
      const boxB = detections[j].xyxy;
      if (computeIoU(boxA, boxB) > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return keep;
}

function computeIoU(boxA, boxB) {
  const x1 = Math.max(boxA[0], boxB[0]);
  const y1 = Math.max(boxA[1], boxB[1]);
  const x2 = Math.min(boxA[2], boxB[2]);
  const y2 = Math.min(boxA[3], boxB[3]);

  if (x2 < x1 || y2 < y1) return 0;

  const intersection = (x2 - x1) * (y2 - y1);
  const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}

// ==================== CANVAS DRAWING ====================
function clearCanvas() {
  if (!mainCanvas) return;
  const ctx = mainCanvas.getContext("2d");
  ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
}

function drawSourceToCanvas(source) {
  if (!mainCanvas || !source) return;
  const { w, h } = getSourceDims(source);

  mainCanvas.width = w;
  mainCanvas.height = h;

  const ctx = mainCanvas.getContext("2d");
  ctx.drawImage(source, 0, 0, w, h);
}

function drawOverlayOnCanvas(status, reason, detections) {
  if (!mainCanvas) return;
  const ctx = mainCanvas.getContext("2d");

  // Draw bounding boxes
  detections.forEach((det, idx) => {
    const [x1, y1, x2, y2] = det.xyxy;
    const label = `${det.class_name} ${(det.conf * 100).toFixed(0)}%`;

    // Box color based on class
    let color;
    if (det.class_name.includes("exposed")) {
      color = "#ef4444"; // Red for violations
    } else {
      color = "#10b981"; // Green for PPE
    }

    // Draw box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw label background
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);

    // Draw label text
    ctx.fillStyle = "white";
    ctx.font = "14px Arial";
    ctx.fillText(label, x1 + 5, y1 - 8);
  });

  // Draw status banner
  const bannerHeight = 40;
  const bannerY = mainCanvas.height - bannerHeight;

  ctx.fillStyle =
    status === "COMPLIANT"
      ? "rgba(16, 185, 129, 0.9)"
      : "rgba(239, 68, 68, 0.9)";
  ctx.fillRect(0, bannerY, mainCanvas.width, bannerHeight);

  ctx.fillStyle = "white";
  ctx.font = "bold 18px Arial";
  ctx.textAlign = "center";
  ctx.fillText(
    `${status === "COMPLIANT" ? "✓" : "✗"} ${status} - ${reason}`,
    mainCanvas.width / 2,
    bannerY + 25,
  );
  ctx.textAlign = "left";
}

// ==================== UI UPDATES ====================
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
}

function updateDetectionsTable(detections) {
  const tbody = document.getElementById("detTableBody");
  if (!tbody) return;

  tbody.innerHTML = "";
  detections.forEach((det, i) => {
    const row = tbody.insertRow();
    row.insertCell(0).textContent = i + 1;
    row.insertCell(1).textContent = det.class_name;
    row.insertCell(2).textContent = det.conf.toFixed(3);
    row.insertCell(3).textContent =
      `[${det.xyxy.map((v) => v.toFixed(0)).join(", ")}]`;
  });
}

function setProgress(percent) {
  document.getElementById("progressText").textContent =
    `${Math.round(percent)}%`;
}

function updateImageStatus(status, reason, detections) {
  const statusMain = document.getElementById("statusMainText");
  statusMain.textContent = `Image: ${status}`;
  statusMain.className = status === "COMPLIANT" ? "compliant" : "non-compliant";

  document.getElementById("reasonText").textContent = `Reason: ${reason}`;

  const classes = detections.map((d) => d.class_name).join(", ");
  document.getElementById("classesText").textContent = classes || "None";
  document.getElementById("countText").textContent = detections.length;

  updateDetectionsTable(detections);
}

function updateVideoStatus(
  statusText,
  reason,
  processed,
  compliant,
  nonCompliant,
  latestStatus,
  progress,
  detections,
) {
  const statusMain = document.getElementById("statusMainText");
  statusMain.textContent = statusText;

  if (latestStatus) {
    statusMain.className =
      latestStatus === "COMPLIANT" ? "compliant" : "non-compliant";
  }

  document.getElementById("reasonText").textContent = reason
    ? `Reason: ${reason}`
    : "";
  document.getElementById("framesProcessedText").textContent = processed;
  document.getElementById("framesCompliantText").textContent = compliant;
  document.getElementById("framesNonCompliantText").textContent = nonCompliant;
  setProgress(progress);

  if (detections && detections.length > 0) {
    const classes = detections.map((d) => d.class_name).join(", ");
    document.getElementById("classesText").textContent = classes;
    document.getElementById("countText").textContent = detections.length;
    updateDetectionsTable(detections);
  }
}

// ==================== FRAME GALLERY ====================
function clearFrameGallery() {
  const gallery = document.getElementById("frameGallery");
  if (gallery) gallery.innerHTML = "";
}

function addFrameThumbnailFromCanvas(sourceCanvas, maxCount = null) {
  const gallery = document.getElementById("frameGallery");
  if (!gallery) return;

  // Remove oldest if at max
  if (maxCount && gallery.children.length >= maxCount) {
    gallery.removeChild(gallery.firstChild);
  }

  const thumb = document.createElement("div");
  thumb.className = "frame-thumb";

  const img = document.createElement("img");
  img.src = sourceCanvas.toDataURL("image/png");
  thumb.appendChild(img);

  // Add label
  const label = document.createElement("div");
  label.className = "frame-label";
  const statusText = document.getElementById("statusMainText").textContent;
  const isCompliant =
    statusText.includes("COMPLIANT") && !statusText.includes("NON");
  label.className += isCompliant ? " compliant" : " non-compliant";
  label.textContent = isCompliant ? "✓ COMPLIANT" : "✗ NON-COMPLIANT";
  thumb.appendChild(label);

  thumb.addEventListener("click", () => openFrameModal(img.src));
  gallery.appendChild(thumb);
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

// ==================== IMAGE PROCESSING ====================
async function handleImageInference(imgElement) {
  if (!session || !imgElement) return;

  drawSourceToCanvas(imgElement);

  const prep = preprocess(imgElement);
  const feeds = {};
  feeds[modelInputName] = prep.tensor;

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const outputTensor = results[outputName];
  console.log("OUTPUT DIMS:", outputTensor.dims);

  const detections = parseDetections(outputTensor, prep);
  currentDetections = detections;

  const classesSet = new Set(detections.map((d) => d.class_name));
  const { status, reason } = decideCompliance(classesSet);

  drawOverlayOnCanvas(status, reason, detections);
  updateImageStatus(status, reason, detections);

  // Send to dashboard
  await sendFrameToDashboard(
    status,
    reason,
    Array.from(classesSet),
    mainCanvas,
    detections,
  );

  // Add to gallery
  addFrameThumbnailFromCanvas(mainCanvas, GALLERY_MAX_ITEMS);

  // Beep if non-compliant
  if (status === "NON-COMPLIANT") {
    playBeep("warning");
  }

  console.log("✓ Image inference complete:", status);
}

// ==================== VIDEO PROCESSING ====================
function stopVideoProcessing() {
  videoProcessing = false;
  if (videoTimerId) {
    clearTimeout(videoTimerId);
    videoTimerId = null;
  }

  const stopBtn = document.getElementById("stopBtn");
  const runBtn = document.getElementById("runCheckBtn");
  if (stopBtn) stopBtn.disabled = true;
  if (runBtn) runBtn.disabled = false;
}

async function processVideoFrame() {
  if (!videoProcessing || !videoElement || !session) return;

  if (videoElement.ended || videoElement.paused) {
    stopVideoProcessing();
    updateVideoStatus(
      "Video complete",
      "",
      videoFramesProcessed,
      videoFramesCompliant,
      videoFramesNonCompliant,
      null,
      100,
      [],
    );
    return;
  }

  // Draw current frame
  drawSourceToCanvas(videoElement);

  // Run inference
  const prep = preprocess(videoElement);
  const feeds = {};
  feeds[modelInputName] = prep.tensor;

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const outputTensor = results[outputName];
  console.log("OUTPUT DIMS:", outputTensor.dims);

  const detections = parseDetections(outputTensor, prep);
  currentDetections = detections;

  const classesSet = new Set(detections.map((d) => d.class_name));
  const { status, reason } = decideCompliance(classesSet);

  // Update stats
  videoFramesProcessed++;
  if (status === "COMPLIANT") {
    videoFramesCompliant++;
  } else if (status === "NON-COMPLIANT") {
    videoFramesNonCompliant++;
    playBeep("warning");
  }

  // Draw overlay
  drawOverlayOnCanvas(status, reason, detections);

  // Update UI
  const progress = (videoElement.currentTime / videoElement.duration) * 100;
  updateVideoStatus(
    `Video frame: ${status}`,
    reason,
    videoFramesProcessed,
    videoFramesCompliant,
    videoFramesNonCompliant,
    status,
    progress,
    detections,
  );

  // Send every 10th frame to dashboard
  if (videoFramesProcessed % 10 === 0) {
    await sendFrameToDashboard(
      status,
      reason,
      Array.from(classesSet),
      mainCanvas,
      detections,
    );
  }

  // Add every 3rd frame to gallery
  if (videoFramesProcessed % 3 === 0) {
    addFrameThumbnailFromCanvas(mainCanvas, GALLERY_MAX_ITEMS);
  }

  // Schedule next frame
  if (videoProcessing) {
    videoTimerId = setTimeout(processVideoFrame, VIDEO_INTERVAL_MS);
  }
}

async function startVideoProcessing() {
  if (!videoElement || !session) return;

  videoFramesProcessed = 0;
  videoFramesCompliant = 0;
  videoFramesNonCompliant = 0;
  videoProcessing = true;

  videoElement.currentTime = 0;
  await videoElement.play();

  const runBtn = document.getElementById("runCheckBtn");
  const stopBtn = document.getElementById("stopBtn");
  if (runBtn) runBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = false;

  updateVideoStatus("Processing video...", "", 0, 0, 0, null, 0, []);
  processVideoFrame();
}

// ==================== FILE HANDLING ====================
async function handleFileChange(event) {
  const file = event.target.files[0];
  if (!file) return;

  stopVideoProcessing();
  clearFrameGallery();
  clearCanvas();
  resetStatusBox();

  videoFramesProcessed = 0;
  videoFramesCompliant = 0;
  videoFramesNonCompliant = 0;

  setFileLoading(true, `Loading ${file.name}...`);

  if (fileObjectUrl) {
    URL.revokeObjectURL(fileObjectUrl);
  }
  fileObjectUrl = URL.createObjectURL(file);

  const fileType = file.type;

  if (fileType.startsWith("image/")) {
    mode = "image";
    imageElement = new Image();
    imageElement.onload = () => {
      setFileLoading(false);
      document.getElementById("statusModeText").textContent =
        `Mode: Image (${file.name})`;
      const runBtn = document.getElementById("runCheckBtn");
      if (runBtn) runBtn.disabled = false;
      console.log("✓ Image loaded:", file.name);
    };
    imageElement.onerror = () => {
      setFileLoading(false);
      alert("Failed to load image.");
    };
    imageElement.src = fileObjectUrl;
  } else if (fileType.startsWith("video/")) {
    mode = "video";
    videoElement = document.getElementById("hiddenVideo");
    videoElement.src = fileObjectUrl;

    videoElement.onloadedmetadata = () => {
      setFileLoading(false);
      document.getElementById("statusModeText").textContent =
        `Mode: Video (${file.name}) - ${videoElement.duration.toFixed(1)}s`;
      const runBtn = document.getElementById("runCheckBtn");
      if (runBtn) runBtn.disabled = false;
      console.log("✓ Video loaded:", file.name);
    };

    videoElement.onerror = () => {
      setFileLoading(false);
      alert("Failed to load video.");
    };
  } else {
    setFileLoading(false);
    alert("Unsupported file type. Please upload an image or video.");
  }
}

// ==================== CONTROL HANDLERS ====================
async function handleRunClick() {
  if (mode === "image" && imageElement) {
    const runBtn = document.getElementById("runCheckBtn");
    if (runBtn) runBtn.disabled = true;

    await handleImageInference(imageElement);

    if (runBtn) runBtn.disabled = false;
    showToast("✓ Image processing complete", 2000);
  } else if (mode === "video" && videoElement) {
    await startVideoProcessing();
  }
}

function handleStopClick() {
  stopVideoProcessing();
  showToast("Processing stopped", 1500);
}

function handleResetClick() {
  stopVideoProcessing();

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

  if (fileInput) {
    fileInput.value = "";
    fileInput.disabled = false;
  }
  if (runBtn) runBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = true;

  mode = null;
  imageElement = null;
  currentDetections = [];
  videoFramesProcessed = 0;
  videoFramesCompliant = 0;
  videoFramesNonCompliant = 0;

  setProgress(0);
  clearCanvas();
  resetStatusBox();
  clearFrameGallery();

  console.log("✓ Reset complete - ready for new file");
}

// ==================== SETUP ====================
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

  if (fileInput) fileInput.addEventListener("change", handleFileChange);
  if (runBtn) runBtn.addEventListener("click", handleRunClick);
  if (stopBtn) stopBtn.addEventListener("click", handleStopClick);
  if (resetBtn) resetBtn.addEventListener("click", handleResetClick);

  if (slider && confLabel) {
    confLabel.textContent = slider.value;
    confThreshold = parseFloat(slider.value);
    slider.addEventListener("input", () => {
      confLabel.textContent = slider.value;
      confThreshold = parseFloat(slider.value);
    });
  }

  setupFrameModal();

  // Initialize model
  initModel().catch((err) => {
    console.error("Error loading model:", err);
    alert("Failed to load model. Check console.");
  });
}

document.addEventListener("DOMContentLoaded", setup);
