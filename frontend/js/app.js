/* frontend/js/app.js */

// Global App State
let activeInputTab = 'upload'; // 'upload' | 'camera'
let cameraStream = null;
let currentFile = null;
let sessionScanCount = 0;

// DOM Selectors
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadTabContent = document.getElementById('upload-tab-content');
const cameraTabContent = document.getElementById('camera-tab-content');
const tabUploadBtn = document.getElementById('tab-upload-btn');
const tabCameraBtn = document.getElementById('tab-camera-btn');
const scannerLineIndicator = document.getElementById('scanner-line-indicator');
const cameraStreamVideo = document.getElementById('camera-stream');

const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsSkeleton = document.getElementById('results-skeleton');
const resultsContent = document.getElementById('results-content');

// --- Input Tab Navigation ---
function switchInputTab(tab) {
  if (activeInputTab === tab) return;
  activeInputTab = tab;

  if (tab === 'upload') {
    tabUploadBtn.classList.add('active');
    tabCameraBtn.classList.remove('active');
    uploadTabContent.style.display = 'flex';
    cameraTabContent.style.display = 'none';
    stopCamera();
  } else {
    tabUploadBtn.classList.remove('active');
    tabCameraBtn.classList.add('active');
    uploadTabContent.style.display = 'none';
    cameraTabContent.style.display = 'flex';
    startCamera();
  }
}

// --- Webcam Controller ---
async function startCamera() {
  stopCamera(); // Make sure previous streams are cleared
  scannerLineIndicator.style.display = 'block';

  try {
    const constraints = {
      video: {
        facingMode: 'environment', // Prefer back camera on mobile
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    cameraStreamVideo.srcObject = cameraStream;
  } catch (err) {
    console.error('Camera access error:', err);
    alert('Unable to access camera. Please allow camera permissions or switch to file upload mode.');
    switchInputTab('upload');
  }
}

function stopCamera() {
  scannerLineIndicator.style.display = 'none';
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
    cameraStreamVideo.srcObject = null;
  }
}

// --- Drag & Drop File Handlers ---
function triggerFileInput() {
  fileInput.click();
}

// Drag over zone state highlight
['dragenter', 'dragover'].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  }, false);
});

['dragleave', 'drop'].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  }, false);
});

// Drop handler
dropZone.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length > 0) {
    processFile(files[0]);
  }
});

function handleFileSelect(e) {
  const files = e.target.files;
  if (files.length > 0) {
    processFile(files[0]);
  }
}

// Client-side validation before network transmission
function processFile(file) {
  const allowedExtensions = ['jpg', 'jpeg', 'png', 'webp'];
  const ext = file.name.split('.').pop().toLowerCase();
  
  if (!allowedExtensions.includes(ext)) {
    alert(`Unsupported file type. Please upload an image with one of these extensions: ${allowedExtensions.join(', ')}`);
    return;
  }

  // 10 MB size check matching server limit
  const maxBytes = 10 * 1024 * 1024;
  if (file.size > maxBytes) {
    alert(`File is too large. Maximum size allowed is 10 MB (Your file is ${(file.size / (1024 * 1024)).toFixed(1)} MB).`);
    return;
  }

  currentFile = file;
  const previewUrl = URL.createObjectURL(file);
  postImageToServer(file, previewUrl);
}

// Camera Capture
function captureAndClassify() {
  if (!cameraStream) return;

  // Create temporary off-screen canvas context
  const canvas = document.createElement('canvas');
  canvas.width = cameraStreamVideo.videoWidth || 640;
  canvas.height = cameraStreamVideo.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  
  // Flip image horizontally to match mirrored preview
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(cameraStreamVideo, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    if (blob) {
      // Re-package as a File object
      const capturedFile = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
      const previewUrl = URL.createObjectURL(blob);
      currentFile = capturedFile;
      postImageToServer(capturedFile, previewUrl);
      
      // Keep UI clean: toggle back to upload tab view
      switchInputTab('upload');
    }
  }, 'image/jpeg', 0.95);
}

// --- API Network Execution ---
async function postImageToServer(fileObject, previewUrl) {
  showState('loading');

  const formData = new FormData();
  formData.append('file', fileObject);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      let errDetail = 'Inference request failed';
      try {
        const errorData = await response.json();
        errDetail = errorData.detail || errDetail;
      } catch (jsonErr) {}
      throw new Error(errDetail);
    }

    const payload = await response.json();
    renderPredictionResults(payload, previewUrl);
  } catch (err) {
    console.error('API execution error:', err);
    alert(`Classification failed: ${err.message}`);
    resetDashboard();
  }
}

// --- DOM Rendering Engine ---
function renderPredictionResults(data, previewUrl) {
  // Update local session count
  sessionScanCount++;
  document.getElementById('session-scan-count').textContent = sessionScanCount;

  // Render preview media image
  document.getElementById('result-preview-image').src = previewUrl;
  
  // Update matched class statistics
  const confidencePercent = Math.round(data.prediction.confidence * 100);
  document.getElementById('result-confidence-text').textContent = `${confidencePercent}% Match`;

  // Basic info labels
  document.getElementById('result-plastic-name').textContent = data.prediction.plastic_type;
  document.getElementById('result-fullname-text').textContent = data.prediction.full_name;
  document.getElementById('result-resin-badge').textContent = data.prediction.resin_code;
  document.getElementById('result-recyclability-text').textContent = data.info.recyclability;
  document.getElementById('result-decomposition-text').textContent = `${data.info.decomposition_years} Years`;
  document.getElementById('result-safety-score').textContent = `${data.info.recyclability_score} / 5`;

  // Uncertainty Box Alert display
  const uncertaintyAlert = document.getElementById('result-uncertainty-alert');
  if (data.prediction.is_uncertain) {
    document.getElementById('result-uncertainty-message').textContent = data.prediction.uncertainty_message;
    uncertaintyAlert.style.display = 'flex';
  } else {
    uncertaintyAlert.style.display = 'none';
  }

  // Draw Probability bars
  const probRowsContainer = document.getElementById('probability-distribution-rows');
  probRowsContainer.innerHTML = '';
  
  // Convert object to sorted array of key-value pairs (descending)
  const sortedProbs = Object.entries(data.prediction.all_probabilities)
    .sort((a, b) => b[1] - a[1]);

  sortedProbs.forEach(([clsName, probVal]) => {
    const isTopClass = clsName === data.prediction.plastic_type;
    const pct = Math.round(probVal * 100);

    const row = document.createElement('div');
    row.className = `probability-bar-row ${isTopClass ? 'top-class' : ''}`;
    row.innerHTML = `
      <div class="probability-label">
        <span>${clsName}</span>
        <span>${pct}%</span>
      </div>
      <div class="probability-track">
        <div class="probability-fill" style="width: ${pct}%"></div>
      </div>
    `;
    probRowsContainer.appendChild(row);
  });

  // Inject detail lists
  injectList('result-uses-list', data.info.common_uses);
  injectList('result-tips-list', data.suggestions.recycling_tips);
  injectList('result-alternatives-list', data.suggestions.eco_alternatives);

  // Safety & Warning Details Text
  const healthTextContainer = document.getElementById('result-health-text');
  healthTextContainer.innerHTML = `<p>${data.info.health_concerns}</p>`;
  if (data.info.warning) {
    healthTextContainer.innerHTML += `
      <div class="alert-box alert-warning" style="margin-top: 1rem; padding: 0.75rem 1rem;">
        <svg class="alert-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        <span><strong>Warning:</strong> ${data.info.warning}</span>
      </div>
    `;
  }

  // Update suggestions source badge styles
  const sourceText = document.getElementById('suggestions-source-text');
  const sourceIndicator = document.getElementById('suggestions-source-indicator');
  
  sourceText.textContent = data.suggestions.source.toUpperCase();
  if (data.suggestions.source === 'ai') {
    sourceIndicator.classList.add('ai');
  } else {
    sourceIndicator.classList.remove('ai');
  }

  // Collapse all accordion blocks except the first one initially
  document.querySelectorAll('.details-card').forEach(card => card.classList.remove('open'));
  document.getElementById('details-card-uses').classList.add('open');

  showState('result');
}

function injectList(elementId, itemsArray) {
  const container = document.getElementById(elementId);
  container.innerHTML = '';
  itemsArray.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    container.appendChild(li);
  });
}

// Tabbed details accordion handler
function toggleDetailsCard(cardId) {
  const card = document.getElementById(`details-card-${cardId}`);
  const isOpen = card.classList.contains('open');
  
  // Collapse all card elements
  document.querySelectorAll('.details-card').forEach(c => c.classList.remove('open'));

  // Toggle clicked card
  if (!isOpen) {
    card.classList.add('open');
  }
}

// Reset view states
function resetDashboard() {
  currentFile = null;
  fileInput.value = '';
  showState('placeholder');
}

// State engine controller
function showState(state) {
  // states: 'placeholder' | 'loading' | 'result'
  if (state === 'placeholder') {
    resultsPlaceholder.style.display = 'flex';
    resultsSkeleton.style.display = 'none';
    resultsContent.style.display = 'none';
  } else if (state === 'loading') {
    resultsPlaceholder.style.display = 'none';
    resultsSkeleton.style.display = 'flex';
    resultsContent.style.display = 'none';
  } else if (state === 'result') {
    resultsPlaceholder.style.display = 'none';
    resultsSkeleton.style.display = 'none';
    resultsContent.style.display = 'flex';
  }
}
