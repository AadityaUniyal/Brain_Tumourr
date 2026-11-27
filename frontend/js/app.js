// ==========================================================================
// NEUROVISION ELITE: Google Antigravity Inspired Client Engine
// ==========================================================================

let currentDiagnostic = null;
let currentRadiogenomic = null;
let selectedFile = null;

// Backend API URL Configuration (Default: Render Cloud Backend)
const DEFAULT_CLOUD_BACKEND = "https://neurovision-elite-backend.onrender.com";
let API_BASE_URL = localStorage.getItem('neurovision_api_url') || DEFAULT_CLOUD_BACKEND;

// DOM Elements
const apiBaseUrlInput = document.getElementById('apiBaseUrlInput');
const currentBackendDisplay = document.getElementById('currentBackendDisplay');
const saveBackendUrlBtn = document.getElementById('saveBackendUrlBtn');
const healthStatusText = document.getElementById('healthStatusText');
const healthDot = document.getElementById('healthDot');

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const fileNameDisplay = document.getElementById('fileName');
const selectedFileInfo = document.getElementById('selectedFileInfo');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const welcomePlaceholder = document.getElementById('welcomePlaceholder');
const resultsView = document.getElementById('resultsView');

// --------------------------------------------------------------------------
// 1. Synaptic Neural Canvas Animation (60FPS Interactive Particle Network)
// --------------------------------------------------------------------------
const canvas = document.getElementById('neuralCanvas');
if (canvas) {
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const nodeCount = Math.min(65, Math.floor((width * height) / 18000));
    const nodes = [];
    const maxDistance = 140;

    let mouse = { x: null, y: null, radius: 160 };
    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });
    window.addEventListener('mouseleave', () => {
        mouse.x = null;
        mouse.y = null;
    });

    class SynapseNode {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.6;
            this.vy = (Math.random() - 0.5) * 0.6;
            this.radius = Math.random() * 2 + 1;
            this.baseAlpha = Math.random() * 0.4 + 0.2;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;

            if (mouse.x !== null && mouse.y !== null) {
                const dx = mouse.x - this.x;
                const dy = mouse.y - this.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < mouse.radius) {
                    const force = (mouse.radius - dist) / mouse.radius;
                    this.x -= (dx / dist) * force * 1.5;
                    this.y -= (dy / dist) * force * 1.5;
                }
            }
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(56, 189, 248, ${this.baseAlpha})`;
            ctx.fill();
        }
    }

    for (let i = 0; i < nodeCount; i++) {
        nodes.push(new SynapseNode());
    }

    function animateSynapses() {
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < nodes.length; i++) {
            nodes[i].update();
            nodes[i].draw();

            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < maxDistance) {
                    const alpha = (1 - dist / maxDistance) * 0.25;
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animateSynapses);
    }
    animateSynapses();
}

// --------------------------------------------------------------------------
// 2. Real-Time Backend Latency & Connectivity Monitor
// --------------------------------------------------------------------------
function updateBackendDisplay() {
    if (API_BASE_URL) {
        currentBackendDisplay.textContent = API_BASE_URL.replace('https://', '').replace('http://', '');
        if (apiBaseUrlInput) apiBaseUrlInput.value = API_BASE_URL;
    } else {
        currentBackendDisplay.textContent = "Auto (Same Origin)";
        if (apiBaseUrlInput) apiBaseUrlInput.value = "";
    }
}
updateBackendDisplay();

async function pingBackendHealth() {
    const targetUrl = API_BASE_URL || window.location.origin;
    const startTime = performance.now();
    try {
        const res = await fetch(`${targetUrl}/health`, { method: 'GET', cache: 'no-cache' });
        const latency = Math.round(performance.now() - startTime);
        if (res.ok) {
            healthStatusText.textContent = `Live: ${latency}ms`;
            healthDot.style.backgroundColor = '#10b981';
            healthDot.style.boxShadow = '0 0 10px #10b981';
        } else {
            throw new Error(`HTTP ${res.status}`);
        }
    } catch (e) {
        healthStatusText.textContent = 'Connecting...';
        healthDot.style.backgroundColor = '#f59e0b';
        healthDot.style.boxShadow = '0 0 10px #f59e0b';
    }
}
pingBackendHealth();
setInterval(pingBackendHealth, 15000);

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
        pingBackendHealth();
        alert("Backend target updated to: " + (API_BASE_URL || "Same Origin"));
    });
}

// --------------------------------------------------------------------------
// 3. File Selection & 1-Click Interactive Clinical Patient Simulator
// --------------------------------------------------------------------------
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
    const sizeKb = (file.size / 1024).toFixed(1);
    document.getElementById('fileSize').textContent = `${sizeKb} KB`;
    selectedFileInfo.classList.remove('d-none');
    analyzeBtn.disabled = false;
}

// 1-Click Patient Scenario Loader
async function loadSampleCase(caseType) {
    const filenameMap = {
        'glioma': 'glioma_sample.png',
        'meningioma': 'meningioma_sample.png',
        'pituitary': 'pituitary_sample.png',
        'ood': 'corrupted_ood_sample.png'
    };

    const targetFilename = filenameMap[caseType] || 'glioma_sample.png';
    const sampleUrl = `samples/${targetFilename}`;

    try {
        const response = await fetch(sampleUrl);
        if (!response.ok) throw new Error("Could not load sample image file");
        const blob = await response.blob();
        const file = new File([blob], targetFilename, { type: 'image/png' });
        
        handleFileSelect(file);

        // Adjust age & demographics per case
        if (caseType === 'glioma') {
            document.getElementById('ageInput').value = 58;
            document.getElementById('sexSelect').value = 'MALE';
        } else if (caseType === 'meningioma') {
            document.getElementById('ageInput').value = 46;
            document.getElementById('sexSelect').value = 'FEMALE';
        } else if (caseType === 'pituitary') {
            document.getElementById('ageInput').value = 39;
            document.getElementById('sexSelect').value = 'FEMALE';
        }

        // Smooth scroll to cockpit and run analysis
        document.getElementById('cockpit').scrollIntoView({ behavior: 'smooth' });
        setTimeout(() => {
            analyzeBtn.click();
        }, 400);
    } catch (err) {
        console.error("Error loading sample case:", err);
        alert("Failed to load sample: " + err.message);
    }
}
window.loadSampleCase = loadSampleCase;

// --------------------------------------------------------------------------
// 4. Dynamic SVG Survival Curve S(t) Renderer
// --------------------------------------------------------------------------
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

        dots += `
            <circle cx="${x}" cy="${y}" r="4" fill="${strokeColor}" stroke="#0b0d14" stroke-width="1.5">
                <title>${pt.month} Months: ${(pt.survival_prob * 100).toFixed(1)}%</title>
            </circle>
        `;
    });

    const gridLines = `
        <line x1="${padX}" y1="${padY}" x2="${padX}" y2="${padY + plotH}" stroke="#334155" stroke-width="1"/>
        <line x1="${padX}" y1="${padY + plotH}" x2="${padX + plotW}" y2="${padY + plotH}" stroke="#334155" stroke-width="1"/>
        <line x1="${padX}" y1="${padY + plotH * 0.5}" x2="${padX + plotW}" y2="${padY + plotH * 0.5}" stroke="#1e293b" stroke-dasharray="3,3"/>
        <text x="${padX - 8}" y="${padY + 4}" fill="#94A3B8" font-size="10" text-anchor="end">1.0</text>
        <text x="${padX - 8}" y="${padY + plotH * 0.5 + 4}" fill="#94A3B8" font-size="10" text-anchor="end">0.5</text>
        <text x="${padX - 8}" y="${padY + plotH + 4}" fill="#94A3B8" font-size="10" text-anchor="end">0.0</text>
        <text x="${padX}" y="${padY + plotH + 18}" fill="#94A3B8" font-size="10" text-anchor="middle">0m</text>
        <text x="${padX + plotW * 0.5}" y="${padY + plotH + 18}" fill="#94A3B8" font-size="10" text-anchor="middle">30m</text>
        <text x="${padX + plotW}" y="${padY + plotH + 18}" fill="#94A3B8" font-size="10" text-anchor="middle">60m</text>
    `;

    svg.innerHTML = `
        <rect width="100%" height="100%" fill="#0a0c13" rx="8"/>
        ${gridLines}
        <path d="${pathD}" fill="none" stroke="${strokeColor}" stroke-width="2.5" stroke-linecap="round"/>
        ${dots}
    `;
}

// --------------------------------------------------------------------------
// 5. Execution Pipeline: Dual-Head Diagnostics + WHO Radiogenomics
// --------------------------------------------------------------------------
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loadingIndicator.classList.remove('d-none');
    welcomePlaceholder.classList.add('d-none');
    resultsView.classList.add('d-none');

    const sliceThickness = parseFloat(document.getElementById('thicknessInput').value) || 3.0;
    const pixelSpacing = parseFloat(document.getElementById('spacingInput').value) || 0.8;
    const patientAge = parseFloat(document.getElementById('ageInput').value) || 54.0;
    const patientSex = document.getElementById('sexSelect').value || 'MALE';

    try {
        // Run Diagnostic Endpoint
        const diagFormData = new FormData();
        diagFormData.append('file', selectedFile);

        const diagPromise = fetch(`${API_BASE_URL}/api/v1/diagnose?slice_thickness_mm=${sliceThickness}&pixel_spacing_mm=${pixelSpacing}`, {
            method: 'POST',
            body: diagFormData
        });

        // Run Radiogenomics Endpoint
        const radFormData = new FormData();
        radFormData.append('file', selectedFile);

        const radPromise = fetch(`${API_BASE_URL}/api/v1/radiogenomics?patient_age=${patientAge}&patient_sex=${patientSex}&is_gbm=false`, {
            method: 'POST',
            body: radFormData
        });

        const [diagRes, radRes] = await Promise.all([diagPromise, radPromise]);

        if (!diagRes.ok) throw new Error(`Diagnostic failed: HTTP ${diagRes.status}`);
        if (!radRes.ok) throw new Error(`Radiogenomics failed: HTTP ${radRes.status}`);

        currentDiagnostic = await diagRes.json();
        currentRadiogenomic = await radRes.json();

        displayResults(currentDiagnostic, currentRadiogenomic);

        // Celebratory confetti if approved
        if (currentDiagnostic.selective_triage && currentDiagnostic.selective_triage.is_selective_approved && window.confetti) {
            window.confetti({ particleCount: 40, spread: 60, origin: { y: 0.7 } });
        }
    } catch (err) {
        console.error("Analysis Pipeline Failure:", err);
        alert(`Analysis Error: ${err.message}. Ensure backend is active.`);
        welcomePlaceholder.classList.remove('d-none');
    } finally {
        analyzeBtn.disabled = false;
        loadingIndicator.classList.add('d-none');
    }
});

function displayResults(diag, rad) {
    // 1. Conformal Risk-Controlled Triage Decision
    const st = diag.selective_triage;
    const banner = document.getElementById('conformalTriageBanner');
    const actionBadge = document.getElementById('triageActionBadge');
    const priorityBadge = document.getElementById('triagePriorityBadge');
    const actionLabel = document.getElementById('triageActionLabel');
    const clinicalGuidance = document.getElementById('triageClinicalGuidance');
    const guaranteeText = document.getElementById('conformalGuaranteeText');

    if (st) {
        if (st.is_selective_approved) {
            banner.className = "glass-card mb-4 border-success";
            actionBadge.className = "badge bg-success fs-6 px-3 py-1";
            actionBadge.innerHTML = '<i class="fa-solid fa-check-circle me-1"></i> AUTOMATED SIGN-OFF';
            priorityBadge.className = "badge bg-dark border border-success text-success small";
            priorityBadge.textContent = "Priority: Routine Validated";
            actionLabel.textContent = "Conformal Finite-Sample Coverage Satisfied";
            clinicalGuidance.textContent = st.clinical_guidance;
            guaranteeText.textContent = `${(st.selective_accuracy_guarantee * 100).toFixed(2)}%`;
        } else {
            const isStat = st.triage_priority && st.triage_priority.includes("STAT");
            banner.className = isStat ? "glass-card mb-4 border-danger" : "glass-card mb-4 border-warning";
            actionBadge.className = isStat ? "badge bg-danger fs-6 px-3 py-1" : "badge bg-warning text-dark fs-6 px-3 py-1";
            actionBadge.innerHTML = isStat ? '<i class="fa-solid fa-triangle-exclamation me-1"></i> STAT NEUROSURGICAL TRIAGE' : '<i class="fa-solid fa-bell me-1"></i> ELEVATED REVIEW';
            priorityBadge.className = isStat ? "badge bg-dark border border-danger text-danger small" : "badge bg-dark border border-warning text-warning small";
            priorityBadge.textContent = `Priority: ${st.triage_priority}`;
            actionLabel.textContent = "Equivocal / High-Risk Profile Detected";
            clinicalGuidance.textContent = st.clinical_guidance;
            guaranteeText.textContent = "89.71%";
        }
    }

    // 2. Primary Diagnostic Summary
    document.getElementById('predictedClass').textContent = diag.diagnostic.predicted_tumor_type;
    document.getElementById('confidenceScore').textContent = `${(diag.diagnostic.confidence * 100).toFixed(1)}%`;
    document.getElementById('tumorVolume').textContent = `${diag.quantification.estimated_volume_cm3.toFixed(2)} cm³`;

    // 3. Tri-Modal Visualizer
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

    // 4. WHO CNS5 2021 Molecular Biopsy
    document.getElementById('whoSubtypeDisplay').textContent = rad.who_2021_subtype;
    document.getElementById('clinicalGuidanceDisplay').textContent = rad.idh_clinical_guidance;

    // IDH
    document.getElementById('idhProbText').textContent = `${(rad.idh_mutation_probability * 100).toFixed(1)}%`;
    document.getElementById('idhStatusText').textContent = rad.idh_status;
    document.getElementById('idhBar').style.width = `${Math.round(rad.idh_mutation_probability * 100)}%`;

    // 1p/19q
    document.getElementById('codelProbText').textContent = `${(rad.codeletion_1p19q_probability * 100).toFixed(1)}%`;
    document.getElementById('codelStatusText').textContent = rad.codeletion_1p19q_status;
    document.getElementById('codelBar').style.width = `${Math.round(rad.codeletion_1p19q_probability * 100)}%`;

    // 5. DeepSurv Survival
    document.getElementById('hazardScoreDisplay').textContent = `g(x) = ${rad.deepsurv_hazard_score.toFixed(3)} log-hazard`;
    document.getElementById('medianSurvivalDisplay').textContent = `${rad.median_overall_survival_months} months`;
    const riskBadge = document.getElementById('riskTierBadge');
    riskBadge.textContent = rad.risk_tier;
    riskBadge.className = rad.risk_tier === "High Risk" ? "badge bg-danger fs-6 px-3 py-1" : "badge bg-success fs-6 px-3 py-1";

    renderSurvivalCurve(rad.survival_curve, rad.risk_tier);

    // 6. SHAP Feature Attributions
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
                    <span class="text-white">${attr.feature_name}</span>
                    <strong class="${colorClass}">${sign}${attr.shap_value.toFixed(3)} log-hazard</strong>
                </div>
                <div class="progress bg-secondary-subtle" style="height: 6px;">
                    <div class="progress-bar ${bgClass}" style="width: ${widthPct}%"></div>
                </div>
            `;
            shapList.appendChild(item);
        });
    }

    resultsView.classList.remove('d-none');
}

// --------------------------------------------------------------------------
// 6. 1-Click Tumor Board Consultation Dossier (PDF) Download
// --------------------------------------------------------------------------
const downloadPdfBtn = document.getElementById('downloadPdfDossierBtn');
if (downloadPdfBtn) {
    downloadPdfBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            alert("Please select or analyze a scan before downloading the clinical dossier.");
            return;
        }

        const originalText = downloadPdfBtn.innerHTML;
        downloadPdfBtn.disabled = true;
        downloadPdfBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-1"></i> Generating PDF Dossier...';

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const ageVal = document.getElementById('ageInput') ? document.getElementById('ageInput').value : 54;
            const sexVal = document.getElementById('sexSelect') ? document.getElementById('sexSelect').value : 'MALE';
            const endpoint = `${API_BASE_URL}/api/v1/export-report?patient_age=${ageVal}&patient_sex=${sexVal}`;

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Server returned HTTP ${response.status}`);

            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = `Tumor_Board_Dossier_${selectedFile.name.replace(/\.[^/.]+$/, "")}.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(downloadUrl);
        } catch (err) {
            console.error("PDF generation failed:", err);
            alert("Failed to generate PDF dossier: " + err.message);
        } finally {
            downloadPdfBtn.disabled = false;
            downloadPdfBtn.innerHTML = originalText;
        }
    });
}
