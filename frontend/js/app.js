// NeuroVision Elite: Client Application Script (WHO 2021 Virtual Biopsy Edition)
let currentDiagnostic = null;
let currentRadiogenomic = null;

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

// Draw Dynamic SVG Survival Curve S(t)
function renderSurvivalCurve(curvePoints, riskTier) {
    const svg = document.getElementById('survivalSvg');
    if (!svg || !curvePoints || curvePoints.length === 0) return;

    const width = 350;
    const height = 240;
    const padX = 40;
    const padY = 25;
    const plotW = width - padX - 20;
    const plotH = height - padY - 30;

    const maxMonth = 60.0;
    const strokeColor = riskTier === "High Risk" ? "#EF4444" : "#10B981";

    let pathD = "";
    let dots = "";

    curvePoints.forEach((pt, idx) => {
        const x = padX + (pt.month / maxMonth) * plotW;
        const y = padY + (1.0 - pt.survival_prob) * plotH;

        if (idx === 0) {
            pathD += `M ${x} ${y}`;
        } else {
            pathD += ` L ${x} ${y}`;
        }
        dots += `<circle cx="${x}" cy="${y}" r="4" fill="${strokeColor}" />`;
    });

    svg.innerHTML = `
        <!-- Axes -->
        <line x1="${padX}" y1="${padY}" x2="${padX}" y2="${padY + plotH}" stroke="#475569" stroke-width="1.5" />
        <line x1="${padX}" y1="${padY + plotH}" x2="${padX + plotW}" y2="${padY + plotH}" stroke="#475569" stroke-width="1.5" />
        
        <!-- Grid lines & labels -->
        <line x1="${padX}" y1="${padY + plotH * 0.5}" x2="${padX + plotW}" y2="${padY + plotH * 0.5}" stroke="#334155" stroke-dasharray="3,3" />
        <text x="${padX - 8}" y="${padY + 5}" fill="#94A3B8" font-size="10" text-anchor="end">1.0</text>
        <text x="${padX - 8}" y="${padY + plotH * 0.5 + 4}" fill="#94A3B8" font-size="10" text-anchor="end">0.5</text>
        <text x="${padX - 8}" y="${padY + plotH + 4}" fill="#94A3B8" font-size="10" text-anchor="end">0.0</text>

        <!-- Curve & Points -->
        <path d="${pathD}" fill="none" stroke="${strokeColor}" stroke-width="3" stroke-linecap="round" />
        ${dots}
    `;
}

// Run Analysis
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loadingIndicator.classList.remove('d-none');
    welcomePlaceholder.classList.add('d-none');
    resultsView.classList.add('d-none');

    const sliceThickness = document.getElementById('sliceThickness').value || 3.0;
    const pixelSpacing = document.getElementById('pixelSpacing').value || 0.8;
    const patientAge = document.getElementById('patientAge').value || 52;
    const patientSex = document.getElementById('patientSex').value || "MALE";

    const base = API_BASE_URL ? API_BASE_URL : "";

    try {
        // 1. Call Primary Diagnostic Endpoint (Segmentation + Classification + Grad-CAM)
        const diagFormData = new FormData();
        diagFormData.append('file', selectedFile);
        const diagUrl = `${base}/api/v1/diagnose?slice_thickness_mm=${sliceThickness}&pixel_spacing_mm=${pixelSpacing}`;
        const diagResp = await fetch(diagUrl, { method: 'POST', body: diagFormData });

        if (!diagResp.ok) throw new Error(`Diagnostic service failed: HTTP ${diagResp.status}`);
        currentDiagnostic = await diagResp.json();

        // 2. Call WHO 2021 Radiogenomics & Survival Endpoint
        const radFormData = new FormData();
        radFormData.append('file', selectedFile);
        const isGbm = currentDiagnostic.diagnostic.predicted_tumor_type === "Glioma";
        const radUrl = `${base}/api/v1/radiogenomics?patient_age=${patientAge}&patient_sex=${patientSex}&is_gbm=${isGbm}`;
        const radResp = await fetch(radUrl, { method: 'POST', body: radFormData });

        if (!radResp.ok) throw new Error(`Radiogenomics service failed: HTTP ${radResp.status}`);
        currentRadiogenomic = await radResp.json();

        // Render Results
        displayResults(currentDiagnostic, currentRadiogenomic);

    } catch (err) {
        alert("Clinical pipeline error: " + err.message);
        welcomePlaceholder.classList.remove('d-none');
    } finally {
        analyzeBtn.disabled = false;
        loadingIndicator.classList.add('d-none');
    }
});

function displayResults(diag, rad) {
    // Top Diagnostics
    document.getElementById('predictedClass').textContent = diag.diagnostic.predicted_tumor_type;
    document.getElementById('confidenceScore').textContent = `${(diag.diagnostic.confidence * 100).toFixed(1)}%`;
    document.getElementById('tumorVolume').textContent = `${diag.quantification.estimated_volume_cm3.toFixed(2)} cm³`;
    document.getElementById('tumorArea').textContent = `Tumor Area: ${diag.quantification.tumor_area_mm2.toFixed(1)} mm²`;

    // Visualizer Image
    const visualizerImg = document.getElementById('visualizerImg');
    visualizerImg.src = `data:image/jpeg;base64,${diag.gradcam_base64}`;

    document.getElementById('btnShowGradCAM').onclick = () => {
        visualizerImg.src = `data:image/jpeg;base64,${diag.gradcam_base64}`;
        document.getElementById('btnShowGradCAM').classList.add('active');
        document.getElementById('btnShowMask').classList.remove('active');
    };

    document.getElementById('btnShowMask').onclick = () => {
        visualizerImg.src = `data:image/jpeg;base64,${diag.mask_base64}`;
        document.getElementById('btnShowMask').classList.add('active');
        document.getElementById('btnShowGradCAM').classList.remove('active');
    };

    // WHO 2021 Molecular Genetics
    document.getElementById('whoSubtypeDisplay').textContent = rad.who_2021_subtype;
    document.getElementById('clinicalGuidanceDisplay').textContent = rad.idh_clinical_guidance;
    
    // Risk Tier Badge
    const riskBadge = document.getElementById('riskTierBadge');
    riskBadge.textContent = rad.risk_tier;
    if (rad.risk_tier === "High Risk") {
        riskBadge.className = "badge bg-danger fs-6 px-3 py-1";
    } else {
        riskBadge.className = "badge bg-success fs-6 px-3 py-1";
    }

    // IDH
    document.getElementById('idhProbText').textContent = `${(rad.idh_mutation_probability * 100).toFixed(1)}%`;
    document.getElementById('idhStatusText').textContent = rad.idh_status;
    document.getElementById('idhBar').style.width = `${(rad.idh_mutation_probability * 100).toFixed(0)}%`;

    // 1p/19q
    document.getElementById('codelProbText').textContent = `${(rad.codeletion_1p19q_probability * 100).toFixed(1)}%`;
    document.getElementById('codelStatusText').textContent = rad.codeletion_1p19q_status;
    document.getElementById('codelBar').style.width = `${(rad.codeletion_1p19q_probability * 100).toFixed(0)}%`;

    // Survival
    document.getElementById('hazardScoreDisplay').textContent = `g(x) = ${rad.deepsurv_hazard_score.toFixed(3)}`;
    document.getElementById('medianSurvivalDisplay').textContent = `${rad.median_overall_survival_months} months`;

    // Render Survival Curve
    renderSurvivalCurve(rad.survival_curve, rad.risk_tier);

    // Extracted Radiomics
    if (rad.radiomic_summary) {
        document.getElementById('radContrast').textContent = rad.radiomic_summary.texture_heterogeneity_glcm;
        document.getElementById('radCompactness').textContent = rad.radiomic_summary.sphericity_compactness;
        document.getElementById('radEntropy').textContent = rad.radiomic_summary.intensity_entropy;
    }

    // Render SHAP Waterfall List
    const shapList = document.getElementById('shapWaterfallList');
    if (shapList && rad.shap_attributions && rad.shap_attributions.length > 0) {
        shapList.innerHTML = '';
        rad.shap_attributions.slice(0, 5).forEach(attr => {
            const isRisk = attr.effect === 'risk_increasing';
            const colorClass = isRisk ? 'text-danger' : 'text-success';
            const bgClass = isRisk ? 'bg-danger' : 'bg-success';
            const sign = attr.shap_value > 0 ? '+' : '';
            const widthPct = Math.min(100, Math.max(10, Math.abs(attr.shap_value) * 120));
            
            const item = document.createElement('div');
            item.className = 'mb-2';
            item.innerHTML = `
                <div class="d-flex justify-content-between small mb-1">
                    <span>${attr.feature_name}</span>
                    <strong class="${colorClass}">${sign}${attr.shap_value.toFixed(3)} log-hazard</strong>
                </div>
                <div class="progress bg-dark" style="height: 6px;">
                    <div class="progress-bar ${bgClass}" style="width: ${widthPct}%"></div>
                </div>
            `;
            shapList.appendChild(item);
        });
    }

    resultsView.classList.remove('d-none');
}
