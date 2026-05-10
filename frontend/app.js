/**
 * Latent Space Navigator — Frontend Application
 *
 * Single-file frontend: WebSocket, MediaPipe hand tracking,
 * DualShock gamepad, input smoothing, radar visualizer.
 */

// ============================================================
// Config
// ============================================================

const WS_PROTO = location.protocol === "https:" ? "wss" : "ws";
const WS_URL = `${WS_PROTO}://${location.host}/ws`;
const NUM_AXES = 6;
const DEFAULT_LABELS = ["style", "texture", "mood", "lighting", "composition", "abstraction"];
const AXIS_COLORS = ["#ff6b8a", "#ff9f43", "#ffd93d", "#6bff8a", "#43c6ff", "#b76bff"];
const DEADZONE = 0.08;
const PAUSE_MS = 1500; // ms of stillness before HQ render
const MOVE_THRESHOLD = 0.02; // coefficient change threshold to count as "moving"

// ============================================================
// DOM refs
// ============================================================

const $ = (sel) => document.querySelector(sel);
const statusBadge    = $("#status-text");
const promptInput    = $("#prompt-input");
const promptSubmit   = $("#prompt-submit");
const axesList       = $("#axes-list");
const genImage       = $("#generated-image");
const genImageBack   = $("#generated-image-back");
const placeholder    = $("#image-placeholder");
const radarCanvas    = $("#radar-canvas");
const perfGen        = $("#perf-gen");
const perfMode       = $("#perf-mode");
const perfRes        = $("#perf-res");
const webcamPip      = $("#webcam-pip");
const webcamVideo    = $("#webcam-video");
const handOverlay    = $("#hand-overlay");
const seedInput      = $("#seed-input");
const scaleSlider    = $("#scale-slider");
const scaleValue     = $("#scale-value");
const smoothSlider   = $("#smooth-slider");
const smoothValue    = $("#smooth-value");
const renderHqBtn    = $("#render-hq-btn");
const settingsToggle = $("#settings-toggle");
const settingsPanel  = $("#settings-panel");
const inputStatus    = $("#input-status");

// ============================================================
// State
// ============================================================

let ws = null;
let connected = false;
let initialized = false;
let inputMode = "sliders"; // "sliders" | "hand" | "gamepad"
let coefficients = new Float64Array(NUM_AXES); // raw from input
let smoothed = new Float64Array(NUM_AXES);     // after LERP
let prevSmoothed = new Float64Array(NUM_AXES);
let axisLabels = [...DEFAULT_LABELS];
let lastMoveTime = 0;
let hqPending = false;
let handLandmarker = null;
let handRunning = false;
let animFrameId = null;

// ============================================================
// WebSocket
// ============================================================

function wsConnect() {
  ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    connected = true;
    setStatus("Connected", true);
  };

  ws.onclose = () => {
    connected = false;
    setStatus("Disconnected", false);
    setTimeout(wsConnect, 2000);
  };

  ws.onerror = () => {
    ws.close();
  };

  ws.onmessage = (evt) => {
    if (evt.data instanceof ArrayBuffer) {
      // Binary = JPEG image
      showImage(evt.data);
    } else {
      const msg = JSON.parse(evt.data);
      handleServerMessage(msg);
    }
  };
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "status":
      setStatus(msg.msg, true);
      break;
    case "axes":
      updateAxesUI(msg.axes);
      break;
    case "perf":
      perfGen.textContent = `${msg.gen_ms} ms`;
      perfMode.textContent = msg.mode;
      perfRes.textContent = msg.mode === "preview" ? "256²" : "1024²";
      break;
  }
}

function setStatus(text, ok) {
  statusBadge.textContent = text;
  statusBadge.classList.toggle("connected", ok);
}

// ============================================================
// Image display (crossfade)
// ============================================================

let imgFlip = false;

function showImage(arrayBuffer) {
  const blob = new Blob([arrayBuffer], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);

  placeholder.classList.add("hidden");

  // Crossfade between two img layers
  const front = imgFlip ? genImageBack : genImage;
  const back  = imgFlip ? genImage : genImageBack;

  front.onload = () => {
    front.style.opacity = "1";
    back.style.opacity = "0";
    // Revoke previous blob URL after transition
    setTimeout(() => {
      if (back.src.startsWith("blob:")) URL.revokeObjectURL(back.src);
    }, 200);
  };
  front.src = url;
  imgFlip = !imgFlip;
}

// ============================================================
// Axes UI
// ============================================================

function updateAxesUI(axes) {
  axesList.innerHTML = "";
  axisLabels = axes.map((a) => a.label);

  axes.forEach((axis, i) => {
    const row = document.createElement("div");
    row.className = "axis-row";

    // Editable label
    const label = document.createElement("span");
    label.className = "axis-label";
    label.textContent = axis.label;
    label.title = `${axis.positive} ↔ ${axis.negative}`;
    label.addEventListener("click", () => startEditLabel(i, label));

    // Slider
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "-3";
    slider.max = "3";
    slider.step = "0.05";
    slider.value = "0";
    slider.id = `axis-slider-${i}`;
    slider.addEventListener("input", () => {
      if (inputMode === "sliders") {
        coefficients[i] = parseFloat(slider.value);
      }
    });

    // Value readout
    const val = document.createElement("span");
    val.className = "axis-value";
    val.id = `axis-val-${i}`;
    val.textContent = "0.00";

    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(val);
    axesList.appendChild(row);
  });

  initialized = true;
}

function startEditLabel(index, labelEl) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "axis-label-input";
  input.value = axisLabels[index];

  const finish = () => {
    const newLabel = input.value.trim();
    if (newLabel && newLabel !== axisLabels[index]) {
      axisLabels[index] = newLabel;
      wsSend({ type: "update_axis", index, label: newLabel });
    }
    // Replace input with label span
    const span = document.createElement("span");
    span.className = "axis-label";
    span.textContent = newLabel || axisLabels[index];
    span.addEventListener("click", () => startEditLabel(index, span));
    input.replaceWith(span);
  };

  input.addEventListener("blur", finish);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") input.blur();
    if (e.key === "Escape") {
      input.value = axisLabels[index];
      input.blur();
    }
  });

  labelEl.replaceWith(input);
  input.focus();
  input.select();
}

// ============================================================
// Input Smoother
// ============================================================

function smoothInputs() {
  const factor = parseFloat(smoothSlider.value) || 0.15;
  for (let i = 0; i < NUM_AXES; i++) {
    smoothed[i] += factor * (coefficients[i] - smoothed[i]);
  }
}

// ============================================================
// Gamepad (DualShock)
// ============================================================

function pollGamepad() {
  if (inputMode !== "gamepad") return;

  const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
  let gp = null;
  for (const g of gamepads) {
    if (g && g.connected) { gp = g; break; }
  }

  if (!gp) {
    inputStatus.textContent = "No controller detected";
    return;
  }

  inputStatus.textContent = `${gp.id.slice(0, 30)}...`;

  // DualShock mapping:
  // Left stick X/Y → axes 0, 1
  // Right stick X/Y → axes 2, 3
  // L2/R2 triggers → axes 4, 5 (mapped from [0,1] to [-1,1])
  const axes = gp.axes;
  const applyDead = (v) => Math.abs(v) < DEADZONE ? 0 : v;
  const scaleMax = parseFloat(scaleSlider.value) || 1.5;

  coefficients[0] = applyDead(axes[0] || 0) * scaleMax;
  coefficients[1] = applyDead(axes[1] || 0) * scaleMax;
  coefficients[2] = applyDead(axes[2] || 0) * scaleMax;
  coefficients[3] = applyDead(axes[3] || 0) * scaleMax;

  // L2/R2 are buttons with analog value
  const l2 = gp.buttons[6] ? gp.buttons[6].value : 0;
  const r2 = gp.buttons[7] ? gp.buttons[7].value : 0;
  coefficients[4] = (l2 * 2 - 1) * scaleMax;
  coefficients[5] = (r2 * 2 - 1) * scaleMax;
}

// Listen for gamepad connect/disconnect
window.addEventListener("gamepadconnected", (e) => {
  console.log("[gamepad] Connected:", e.gamepad.id);
});

// ============================================================
// Hand Tracker (MediaPipe)
// ============================================================

async function initHandTracker() {
  inputStatus.textContent = "Loading hand tracker...";

  const { FilesetResolver, HandLandmarker } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs"
  );

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });

  // Start webcam
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 320, height: 240, facingMode: "user" },
  });
  webcamVideo.srcObject = stream;
  webcamPip.classList.remove("hidden");

  handRunning = true;
  inputStatus.textContent = "Hand tracking active";
  detectHand();
}

function detectHand() {
  if (!handRunning || !handLandmarker) return;

  if (webcamVideo.readyState >= 2) {
    const results = handLandmarker.detectForVideo(webcamVideo, performance.now());

    // Draw overlay
    const ctx = handOverlay.getContext("2d");
    handOverlay.width = webcamVideo.videoWidth || 320;
    handOverlay.height = webcamVideo.videoHeight || 240;
    ctx.clearRect(0, 0, handOverlay.width, handOverlay.height);

    if (results.landmarks && results.landmarks.length > 0) {
      const lm = results.landmarks[0]; // first hand
      const wlm = results.worldLandmarks ? results.worldLandmarks[0] : null;

      // Draw skeleton
      drawHandSkeleton(ctx, lm, handOverlay.width, handOverlay.height);

      // Extract 6DOF
      const scaleMax = parseFloat(scaleSlider.value) || 1.5;

      // 1. Palm center X (normalized -1 to 1)
      const palmX = (lm[0].x + lm[5].x + lm[17].x) / 3;
      coefficients[0] = (palmX - 0.5) * 2 * scaleMax;

      // 2. Palm center Y (normalized -1 to 1, inverted)
      const palmY = (lm[0].y + lm[5].y + lm[17].y) / 3;
      coefficients[1] = -(palmY - 0.5) * 2 * scaleMax;

      // 3. Depth Z
      const palmZ = (lm[0].z + lm[5].z + lm[17].z) / 3;
      coefficients[2] = palmZ * 20 * scaleMax; // z is small, amplify

      // 4. Finger pitch: angle of middle finger relative to wrist
      const pitch = Math.atan2(lm[12].y - lm[0].y, lm[12].x - lm[0].x);
      coefficients[3] = (pitch / Math.PI) * scaleMax;

      // 5. Hand spread: distance between index tip and pinky tip
      const spread = Math.hypot(lm[8].x - lm[20].x, lm[8].y - lm[20].y);
      coefficients[4] = (spread - 0.15) * 8 * scaleMax; // center around typical spread

      // 6. Pinch: thumb tip to index tip distance
      const pinch = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y);
      coefficients[5] = (0.1 - pinch) * 10 * scaleMax; // invert: pinch = positive
    }
  }

  requestAnimationFrame(detectHand);
}

function drawHandSkeleton(ctx, landmarks, w, h) {
  const connections = [
    [0,1],[1,2],[2,3],[3,4],       // thumb
    [0,5],[5,6],[6,7],[7,8],       // index
    [5,9],[9,10],[10,11],[11,12],   // middle
    [9,13],[13,14],[14,15],[15,16], // ring
    [13,17],[17,18],[18,19],[19,20],// pinky
    [0,17],                        // palm base
  ];

  ctx.strokeStyle = "rgba(0, 229, 255, 0.7)";
  ctx.lineWidth = 2;

  for (const [a, b] of connections) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(0, 229, 255, 0.9)";
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function stopHandTracker() {
  handRunning = false;
  webcamPip.classList.add("hidden");
  if (webcamVideo.srcObject) {
    webcamVideo.srcObject.getTracks().forEach((t) => t.stop());
    webcamVideo.srcObject = null;
  }
}

// ============================================================
// Radar Visualizer
// ============================================================

function drawRadar() {
  const ctx = radarCanvas.getContext("2d");
  const W = radarCanvas.width;
  const H = radarCanvas.height;
  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(cx, cy) - 20;

  ctx.clearRect(0, 0, W, H);

  // Grid rings
  for (let r = 1; r <= 3; r++) {
    const radius = (r / 3) * R;
    ctx.beginPath();
    ctx.strokeStyle = `rgba(100, 100, 180, ${0.08 + r * 0.03})`;
    ctx.lineWidth = 1;
    for (let i = 0; i <= NUM_AXES; i++) {
      const angle = (Math.PI * 2 * i) / NUM_AXES - Math.PI / 2;
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  // Axis lines + labels
  for (let i = 0; i < NUM_AXES; i++) {
    const angle = (Math.PI * 2 * i) / NUM_AXES - Math.PI / 2;
    const x = cx + Math.cos(angle) * R;
    const y = cy + Math.sin(angle) * R;

    ctx.beginPath();
    ctx.strokeStyle = "rgba(100, 100, 180, 0.1)";
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();

    // Label
    const lx = cx + Math.cos(angle) * (R + 14);
    const ly = cy + Math.sin(angle) * (R + 14);
    ctx.fillStyle = AXIS_COLORS[i];
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(axisLabels[i]?.slice(0, 6) || "", lx, ly);
  }

  // Value polygon
  const scaleMax = parseFloat(scaleSlider.value) || 1.5;
  ctx.beginPath();
  for (let i = 0; i < NUM_AXES; i++) {
    const angle = (Math.PI * 2 * i) / NUM_AXES - Math.PI / 2;
    const val = Math.min(Math.abs(smoothed[i]) / scaleMax, 1);
    const sign = smoothed[i] >= 0 ? 1 : -0.3; // negative = smaller
    const r = val * R * (sign > 0 ? sign : 0.3);
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(0, 229, 255, 0.12)";
  ctx.fill();
  ctx.strokeStyle = "rgba(0, 229, 255, 0.6)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Dots on polygon vertices
  for (let i = 0; i < NUM_AXES; i++) {
    const angle = (Math.PI * 2 * i) / NUM_AXES - Math.PI / 2;
    const val = Math.min(Math.abs(smoothed[i]) / scaleMax, 1);
    const sign = smoothed[i] >= 0 ? 1 : 0.3;
    const r = val * R * sign;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = AXIS_COLORS[i];
    ctx.fill();
  }
}

// ============================================================
// Main Loop
// ============================================================

function mainLoop() {
  // Poll inputs
  pollGamepad();

  // Smooth
  smoothInputs();

  // Update slider UI to reflect current values (when not in slider mode)
  for (let i = 0; i < NUM_AXES; i++) {
    const slider = document.getElementById(`axis-slider-${i}`);
    const valEl = document.getElementById(`axis-val-${i}`);
    if (slider && inputMode !== "sliders") {
      slider.value = coefficients[i];
    }
    if (valEl) {
      valEl.textContent = smoothed[i].toFixed(2);
    }
  }

  // Draw radar
  drawRadar();

  // Check if moving
  let moving = false;
  for (let i = 0; i < NUM_AXES; i++) {
    if (Math.abs(smoothed[i] - prevSmoothed[i]) > MOVE_THRESHOLD) {
      moving = true;
      break;
    }
  }

  if (moving) {
    lastMoveTime = Date.now();
    hqPending = true;
  }

  // Send coefficients if initialized and any non-zero
  if (initialized && connected) {
    const hasSignal = smoothed.some((v) => Math.abs(v) > 0.01);
    if (moving || hasSignal) {
      wsSend({ type: "move", coefficients: Array.from(smoothed) });
    }

    // Auto HQ render on pause
    if (hqPending && Date.now() - lastMoveTime > PAUSE_MS) {
      hqPending = false;
      wsSend({ type: "render_hq" });
    }
  }

  prevSmoothed.set(smoothed);
  animFrameId = requestAnimationFrame(mainLoop);
}

// ============================================================
// Event Wiring
// ============================================================

// Prompt submit
promptSubmit.addEventListener("click", () => {
  const prompt = promptInput.value.trim();
  if (prompt) {
    wsSend({ type: "init", prompt });
    // Reset coefficients
    coefficients.fill(0);
    smoothed.fill(0);
  }
});

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") promptSubmit.click();
});

// Settings
settingsToggle.addEventListener("click", () => {
  settingsPanel.classList.toggle("hidden");
});

scaleSlider.addEventListener("input", () => {
  scaleValue.textContent = parseFloat(scaleSlider.value).toFixed(1);
  wsSend({ type: "set_scale", scale: parseFloat(scaleSlider.value) });
});

smoothSlider.addEventListener("input", () => {
  smoothValue.textContent = parseFloat(smoothSlider.value).toFixed(2);
});

seedInput.addEventListener("change", () => {
  wsSend({ type: "set_seed", seed: parseInt(seedInput.value) || 42 });
});

renderHqBtn.addEventListener("click", () => {
  wsSend({ type: "render_hq" });
});

// Input mode buttons
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.mode;
    setInputMode(mode);
  });
});

function setInputMode(mode) {
  // Deactivate previous
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
  document.querySelector(`[data-mode="${mode}"]`)?.classList.add("active");

  // Stop hand tracker if switching away
  if (inputMode === "hand" && mode !== "hand") {
    stopHandTracker();
  }

  inputMode = mode;
  coefficients.fill(0);

  switch (mode) {
    case "sliders":
      inputStatus.textContent = "Using manual sliders";
      break;
    case "hand":
      initHandTracker().catch((err) => {
        console.error("[hand] Init failed:", err);
        inputStatus.textContent = `Hand tracking error: ${err.message}`;
        setInputMode("sliders");
      });
      break;
    case "gamepad":
      inputStatus.textContent = "Waiting for controller...";
      break;
  }
}

// ============================================================
// Boot
// ============================================================

wsConnect();
mainLoop();
