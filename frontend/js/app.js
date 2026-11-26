// NeuroVision Elite: Client Application Script
let currentResult = null;

// Backend API URL Configuration (Default: Render cloud backend)
const DEFAULT_CLOUD_BACKEND = "https://neurovision-elite-backend.onrender.com";
let API_BASE_URL = localStorage.getItem('neurovision_api_url') || DEFAULT_CLOUD_BACKEND;
const apiBaseUrlInput = document.getElementById('apiBaseUrlInput');
const currentBackendDisplay = document.getElementById('currentBackendDisplay');
const saveBackendUrlBtn = document.getElementById('saveBackendUrlBtn');

function updateBackendDisplay() {
    if (API_BASE_URL) {
        currentBackendDisplay.textContent = API_BASE_URL.replace('https://', '').replace('http://', '');
        apiBaseUrlInput.value = API_BASE_URL;
    } else {
        currentBackendDisplay.textContent = "Auto (Same Origin)";
        apiBaseUrlInput.value = "";
    }
}
updateBackendDisplay();

if (saveBackendUrlBtn) {
    saveBackendUrlBtn.addEventListener('click', () => {
        let url = apiBaseUrlInput.value.trim().replace(/\/$/, "");
        if (url) {
            localStorage.setItem('neurovision_api_url', url);
            API_BASE_URL = url;
        } else {
            localStorage.removeItem('neurovision_api_url');
            API_BASE_URL = '';
        }
        updateBackendDisplay();
        alert("Backend target updated to: " + (API_BASE_URL || "Same Origin"));
    });
}

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const fileNameDisplay = document.getElementById('fileName');
const selectedFileInfo = document.getElementById('selectedFileInfo');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const welcomePlaceholder = document.getElementById('welcomePlaceholder');
const resultsView = document.getElementById('resultsView');

let selectedFile = null;

// File Selection Handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Drag and drop listeners
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

function handleFileSelect(file) {
    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    selectedFileInfo.classList.remove('d-none');
    analyzeBtn.disabled = false;
}

// Run Analysis
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loadingIndicator.classList.remove('d-none');
    welcomePlaceholder.classList.add('d-none');
    resultsView.classList.add('d-none');

    const formData = new FormData();
    formData.append('file', selectedFile);

    const sliceThickness = document.getElementById('sliceThickness').value || 3.0;
    const pixelSpacing = document.getElementById('pixelSpacing').value || 0.8;

    try {
        const targetUrl = API_BASE_URL ? `${API_BASE_URL}/api/v1/diagnose` : `/api/v1/diagnose`;
        const response = await fetch(`${targetUrl}?slice_thickness_mm=${sliceThickness}&pixel_spacing_mm=${pixelSpacing}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned error: ${response.statusText}`);
        }

        const data = await response.json();
        currentResult = data;
        displayResults(data);
    } catch (err) {
        alert("Clinical Diagnostic Pipeline Error: " + err.message);
        welcomePlaceholder.classList.remove('d-none');
    } finally {
        analyzeBtn.disabled = false;
        loadingIndicator.classList.add('d-none');
    }
});

function displayResults(data) {
    resultsView.classList.remove('d-none');

    // Diagnosis & Confidence
    const diag = data.diagnostic;
    document.getElementById('predictedClass').textContent = diag.predicted_tumor_type;
    document.getElementById('confidenceScore').textContent = (diag.confidence * 100).toFixed(1) + '%';

    // Volumetric quantification
    const quant = data.quantification;
    document.getElementById('tumorVolume').textContent = quant.estimated_volume_cm3 + ' cm³';
    document.getElementById('tumorArea').textContent = `Tumor Area: ${quant.tumor_area_mm2} mm² (${quant.tumor_pixel_count} px)`;

    // OOD & Safety Check
    const oodBadge = document.getElementById('oodBadge');
    if (data.is_ood) {
        oodBadge.className = 'badge bg-danger p-2 my-1 fs-6 text-start d-block';
        oodBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation me-1"></i> OUT-OF-DISTRIBUTION (REJECTED)`;
    } else {
        oodBadge.className = 'badge bg-success p-2 my-1 fs-6 text-start d-block';
        oodBadge.innerHTML = `<i class="fa-solid fa-shield-check me-1"></i> Scan Validated (Energy: ${data.energy_score.toFixed(2)})`;
    }

    // Conformal Prediction Set
    document.getElementById('conformalSet').textContent = `95% Coverage Set: [${diag.conformal_prediction_set.join(', ')}]`;

    // Probability Bars
    const probs = diag.class_probabilities;
    updateBar('Glioma', probs.Glioma);
    updateBar('Meningioma', probs.Meningioma);
    updateBar('Pituitary', probs.Pituitary);

    // Initial Image: Grad-CAM overlay
    showGradCAM();
}

function updateBar(className, prob) {
    const pct = (prob * 100).toFixed(1) + '%';
    document.getElementById(`prob${className}`).textContent = pct;
    document.getElementById(`bar${className}`).style.width = pct;
}

function showGradCAM() {
    if (!currentResult || !currentResult.gradcam_base64) return;
    document.getElementById('visualizerImg').src = 'data:image/jpeg;base64,' + currentResult.gradcam_base64;
    document.getElementById('btnShowGradCAM').classList.add('active');
    document.getElementById('btnShowMask').classList.remove('active');
}

function showMask() {
    if (!currentResult || !currentResult.mask_base64) return;
    document.getElementById('visualizerImg').src = 'data:image/jpeg;base64,' + currentResult.mask_base64;
    document.getElementById('btnShowMask').classList.add('active');
    document.getElementById('btnShowGradCAM').classList.remove('active');
}

document.getElementById('btnShowGradCAM').addEventListener('click', showGradCAM);
document.getElementById('btnShowMask').addEventListener('click', showMask);
