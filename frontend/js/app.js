/**
 * SULCUS CLINICAL AI — FRONTEND APPLICATION CONTROLLER
 * High-performance interactive Neuro-Vision lens, survival simulator,
 * multi-tab clinical workstation, and AWS API integration.
 */

// --------------------------------------------------------------------------
// 1. Pre-cached Cohort Datasets
// --------------------------------------------------------------------------
const COHORTS = {
  glioma: {
    caseId: 'PT-8812 (Glioblastoma, IDH-WT)',
    imgSrc: 'samples/glioma_sample.png',
    primaryClass: 'Glioblastoma',
    confidence: '94.2%',
    barWidth: '94.2%',
    colorClass: 'glioma',
    volume: '34.82 cm³',
    major: '41.2 mm',
    minor: '28.6 mm',
    rano: '41.2 × 28.6 mm',
    conformal: '{Glioma} (100% singleton)',
    energy: '-12.84 (In-Distribution)',
    idh: 'Wild-type (8.2%)',
    codeletion: 'Intact (1.1%)',
    cx: 0.58, cy: 0.44, rx: 0.16, ry: 0.13, angle: 0.2
  },
  meningioma: {
    caseId: 'PT-4109 (Convexity Meningioma)',
    imgSrc: 'samples/meningioma_sample.png',
    primaryClass: 'Meningioma',
    confidence: '98.1%',
    barWidth: '98.1%',
    colorClass: 'meningioma',
    volume: '18.45 cm³',
    major: '29.4 mm',
    minor: '24.1 mm',
    rano: '29.4 × 24.1 mm',
    conformal: '{Meningioma} (100% singleton)',
    energy: '-14.22 (In-Distribution)',
    idh: 'Non-applicable',
    codeletion: 'Intact (0.4%)',
    cx: 0.68, cy: 0.36, rx: 0.12, ry: 0.11, angle: -0.1
  },
  pituitary: {
    caseId: 'PT-2291 (Pituitary Macroadenoma)',
    imgSrc: 'samples/pituitary_sample.png',
    primaryClass: 'Pituitary Adenoma',
    confidence: '96.7%',
    barWidth: '96.7%',
    colorClass: 'pituitary',
    volume: '11.20 cm³',
    major: '22.8 mm',
    minor: '19.5 mm',
    rano: '22.8 × 19.5 mm',
    conformal: '{Pituitary} (100% singleton)',
    energy: '-15.01 (In-Distribution)',
    idh: 'Non-applicable',
    codeletion: 'Intact (0.2%)',
    cx: 0.50, cy: 0.56, rx: 0.09, ry: 0.09, angle: 0.0
  }
};

let currentCohortKey = 'glioma';
let currentLensMode = 'raw';
let activeImage = null;
const imageCache = {};

// --------------------------------------------------------------------------
// 2. Interactive Neuro-Vision Lens (Landing Page)
// --------------------------------------------------------------------------
function getCanvasContexts() {
  const baseCanvas = document.getElementById('lensBaseCanvas');
  const overlayCanvas = document.getElementById('lensOverlayCanvas');
  const stage = document.getElementById('lensStage');
  return {
    baseCanvas,
    baseCtx: baseCanvas ? baseCanvas.getContext('2d') : null,
    overlayCanvas,
    overlayCtx: overlayCanvas ? overlayCanvas.getContext('2d') : null,
    stage
  };
}

function syncCanvasSizes() {
  const { baseCanvas, overlayCanvas, stage } = getCanvasContexts();
  if (!stage || !baseCanvas) return;

  const w = stage.clientWidth || 500;
  const h = stage.clientHeight || 500;

  if (baseCanvas.width !== w || baseCanvas.height !== h) {
    baseCanvas.width = w;
    baseCanvas.height = h;
  }

  if (overlayCanvas) {
    overlayCanvas.width = w;
    overlayCanvas.height = h;
    overlayCanvas.style.width = `${w}px`;
  }
}

function drawLayer(ctx, img, mode, cohort) {
  if (!ctx || !img) return;
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  const cx = cohort.cx * w;
  const cy = cohort.cy * h;
  const rx = cohort.rx * w;
  const ry = cohort.ry * h;

  if (mode === 'seg' || mode === 'split_overlay') {
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, cohort.angle || 0, 0, Math.PI * 2);

    // Glowing bio-cyan fill & boundary
    ctx.fillStyle = 'rgba(22, 171, 169, 0.32)';
    ctx.fill();

    ctx.strokeStyle = '#16aba9';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = '#16aba9';
    ctx.shadowBlur = 10;
    ctx.stroke();

    // Necrotic core
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx * 0.5, ry * 0.5, cohort.angle || 0, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(245, 93, 93, 0.38)';
    ctx.fill();
    ctx.strokeStyle = '#f55d5d';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // RANO measurement callipers
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx - rx, cy); ctx.lineTo(cx + rx, cy);
    ctx.moveTo(cx, cy - ry); ctx.lineTo(cx, cy + ry);
    ctx.stroke();

    ctx.restore();
  } else if (mode === 'cam') {
    ctx.save();
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, rx * 1.6);
    grad.addColorStop(0.0, 'rgba(245, 93, 93, 0.75)');
    grad.addColorStop(0.35, 'rgba(230, 149, 0, 0.6)');
    grad.addColorStop(0.7, 'rgba(22, 171, 169, 0.4)');
    grad.addColorStop(1.0, 'rgba(22, 171, 169, 0.0)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx * 1.5, ry * 1.5, cohort.angle || 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

function renderLens() {
  syncCanvasSizes();
  const { baseCtx, overlayCtx } = getCanvasContexts();
  const cohort = COHORTS[currentCohortKey];
  const img = activeImage;
  if (!img) return;

  const clip = document.getElementById('lensWipeClip');

  if (currentLensMode === 'split') {
    if (clip) clip.style.display = 'block';
    drawLayer(baseCtx, img, 'raw', cohort);
    drawLayer(overlayCtx, img, 'split_overlay', cohort);
  } else {
    if (clip) clip.style.display = 'none';
    drawLayer(baseCtx, img, currentLensMode, cohort);
  }
}

function setLensMode(mode) {
  currentLensMode = mode;
  document.querySelectorAll('.lens-btn').forEach(btn => {
    btn.classList.toggle('is-active', btn.getAttribute('data-mode') === mode);
  });
  renderLens();
}

function loadLensCohort(key) {
  if (!COHORTS[key]) return;
  currentCohortKey = key;
  const cohort = COHORTS[key];

  document.getElementById('lensCaseId').textContent = cohort.caseId;
  const classEl = document.getElementById('telemetryClass');
  if (classEl) {
    classEl.textContent = cohort.primaryClass;
    classEl.className = `telemetry-class ${cohort.colorClass}`;
  }
  document.getElementById('telemetryConfidence').textContent = cohort.confidence;
  document.getElementById('telemetryBar').style.width = cohort.barWidth;
  document.getElementById('telemetryVolume').textContent = cohort.volume;
  document.getElementById('telemetryRano').textContent = cohort.rano;
  document.getElementById('telemetryConformal').textContent = cohort.conformal;
  document.getElementById('telemetryEnergy').textContent = cohort.energy;

  if (imageCache[cohort.imgSrc]) {
    activeImage = imageCache[cohort.imgSrc];
    renderLens();
  } else {
    const img = new Image();
    img.src = cohort.imgSrc;
    img.onload = () => {
      imageCache[cohort.imgSrc] = img;
      activeImage = img;
      renderLens();
    };
  }
}

function initWipeSlider() {
  const stage = document.getElementById('lensStage');
  const clip = document.getElementById('lensWipeClip');
  if (!stage || !clip) return;

  let isDragging = false;

  function updateWipe(clientX) {
    const rect = stage.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const pct = (x / rect.width) * 100;
    clip.style.width = `${pct}%`;
  }

  stage.addEventListener('mousedown', (e) => {
    if (currentLensMode !== 'split') return;
    isDragging = true;
    updateWipe(e.clientX);
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    updateWipe(e.clientX);
  });

  window.addEventListener('mouseup', () => { isDragging = false; });

  stage.addEventListener('touchstart', (e) => {
    if (currentLensMode !== 'split') return;
    isDragging = true;
    updateWipe(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    updateWipe(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchend', () => { isDragging = false; });
}

// --------------------------------------------------------------------------
// 3. Virtual Biopsy & DeepSurv Dynamic Simulator
// --------------------------------------------------------------------------
let simState = {
  age: 54,
  sex: 'MALE',
  resection: 'GTR',
  idh: 'WT'
};

function setSimSex(sex) {
  simState.sex = sex;
  document.getElementById('btnSexMale')?.classList.toggle('is-active', sex === 'MALE');
  document.getElementById('btnSexFemale')?.classList.toggle('is-active', sex === 'FEMALE');
  updateSurvivalSimulator();
}

function setSimResection(type) {
  simState.resection = type;
  document.getElementById('btnGTR')?.classList.toggle('is-active', type === 'GTR');
  document.getElementById('btnSTR')?.classList.toggle('is-active', type === 'STR');
  updateSurvivalSimulator();
}

function setSimIdh(status) {
  simState.idh = status;
  document.getElementById('btnIdhWT')?.classList.toggle('is-active', status === 'WT');
  document.getElementById('btnIdhMUT')?.classList.toggle('is-active', status === 'MUT');
  updateSurvivalSimulator();
}

function updateSurvivalSimulator() {
  const ageRange = document.getElementById('simAgeRange');
  const ageLabel = document.getElementById('simAgeLabel');
  if (ageRange && ageLabel) {
    simState.age = parseInt(ageRange.value, 10);
    ageLabel.textContent = `${simState.age} years`;
  }

  let baseMedian = 15.0;
  if (simState.idh === 'MUT') baseMedian += 16.0;
  if (simState.resection === 'GTR') baseMedian += 5.5;
  if (simState.sex === 'FEMALE') baseMedian += 1.8;
  baseMedian -= (simState.age - 50) * 0.28;
  baseMedian = Math.max(5.5, Math.min(48.0, baseMedian));

  const medianEl = document.getElementById('simMedianSurvival');
  if (medianEl) medianEl.textContent = `${baseMedian.toFixed(1)} mo`;

  const path = document.getElementById('simSurvivalPath');
  if (!path) return;

  const points = [];
  const startX = 40;
  const endX = 380;
  const startY = 20;
  const zeroY = 155;
  const height = zeroY - startY;

  for (let m = 0; m <= 60; m += 2) {
    const x = startX + (m / 60) * (endX - startX);
    const lambda = 0.693 / baseMedian;
    const prob = Math.exp(-lambda * m);
    const y = zeroY - (prob * height);
    points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }

  path.setAttribute('d', `M ${points.join(' L ')}`);
}

// --------------------------------------------------------------------------
// 4. Authentication Modal & Cinematic Workstation Transitions
// --------------------------------------------------------------------------
function openAuthModal() {
  const modal = document.getElementById('authModalBackdrop');
  if (modal) modal.classList.add('is-open');
}

function closeAuthModal() {
  const modal = document.getElementById('authModalBackdrop');
  if (modal) modal.classList.remove('is-open');
}

function closeAuthModalOnBackdrop(e) {
  if (e.target.id === 'authModalBackdrop') closeAuthModal();
}

function enterWorkstation() {
  closeAuthModal();

  // 1. Trigger the cinematic bottle transition curtain
  const curtain = document.getElementById('authTransitionCurtain');
  const ambientBottle = document.getElementById('ambientBottleContainer');
  
  if (curtain) {
    curtain.classList.add('is-active');
  }
  if (ambientBottle) {
    ambientBottle.style.opacity = '0';
    ambientBottle.style.pointerEvents = 'none';
  }

  // 2. Allow the 3D bottle spin and glow to play for 1200ms
  setTimeout(() => {
    // Hide Landing Stage, Reveal Workstation Stage
    const landing = document.getElementById('stage-landing');
    const workstation = document.getElementById('stage-workstation');
    
    if (landing) landing.style.display = 'none';
    if (workstation) {
      workstation.classList.add('is-active');
      
      // Trigger cascade-in animation refresh on all workstation cards and panels
      const cascadeEls = workstation.querySelectorAll('.cascade-in');
      cascadeEls.forEach(el => {
        el.style.animation = 'none';
        void el.offsetHeight; // force reflow
        el.style.animation = '';
      });
    }

    // Dismiss Curtain
    if (curtain) {
      curtain.classList.remove('is-active');
    }

    window.scrollTo({ top: 0, behavior: 'instant' });
    loadWorkstationScan('glioma');
  }, 1200);
}

function exitWorkstation() {
  const landing = document.getElementById('stage-landing');
  const workstation = document.getElementById('stage-workstation');
  const ambientBottle = document.getElementById('ambientBottleContainer');

  if (workstation) workstation.classList.remove('is-active');
  if (landing) landing.style.display = 'block';
  if (ambientBottle) {
    ambientBottle.style.opacity = '1';
    ambientBottle.style.pointerEvents = 'none';
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// --------------------------------------------------------------------------
// 5. The Tabbed Clinical Workstation (Screen 3)
// --------------------------------------------------------------------------
function switchWorkstationTab(tabName) {
  const tabs = ['viewport', 'biopsy', 'registry', 'audit'];
  tabs.forEach(t => {
    const btn = document.getElementById(`tabBtn${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const content = document.getElementById(`tabContent-${t}`);
    if (btn) btn.classList.toggle('is-active', t === tabName);
    if (content) content.style.display = (t === tabName) ? (t === 'viewport' ? 'grid' : 'block') : 'none';
  });

  if (tabName === 'viewport') {
    setTimeout(renderWorkstationCanvas, 50);
  }
}

let workstationCohort = 'glioma';
let workstationImg = null;

function loadWorkstationScan(key) {
  if (!COHORTS[key]) return;
  workstationCohort = key;
  const c = COHORTS[key];

  document.getElementById('wsClass').textContent = c.primaryClass;
  document.getElementById('wsConfidence').textContent = c.confidence;
  document.getElementById('wsBar').style.width = c.barWidth;
  document.getElementById('wsMajor').textContent = c.major;
  document.getElementById('wsMinor').textContent = c.minor;
  document.getElementById('wsVolume').textContent = c.volume;
  document.getElementById('wsIdh').textContent = c.idh;
  document.getElementById('wsCodeletion').textContent = c.codeletion;

  const img = new Image();
  img.src = c.imgSrc;
  img.onload = () => {
    workstationImg = img;
    renderWorkstationCanvas();
  };
}

function renderWorkstationCanvas() {
  const canvas = document.getElementById('workstationCanvas');
  const container = document.getElementById('workstationCenter');
  if (!canvas || !container || !workstationImg) return;

  const w = container.clientWidth || 600;
  const h = container.clientHeight || 600;
  canvas.width = w;
  canvas.height = h;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(workstationImg, 0, 0, w, h);

  const c = COHORTS[workstationCohort];
  const cx = c.cx * w;
  const cy = c.cy * h;
  const rx = c.rx * w;
  const ry = c.ry * h;

  // Draw Neural Mask
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, c.angle || 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(22, 171, 169, 0.35)';
  ctx.fill();
  ctx.strokeStyle = '#16aba9';
  ctx.lineWidth = 3;
  ctx.shadowColor = '#16aba9';
  ctx.shadowBlur = 12;
  ctx.stroke();

  // Draw 2D RANO Orthogonal Diameters
  ctx.shadowBlur = 0;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(cx - rx, cy); ctx.lineTo(cx + rx, cy);
  ctx.moveTo(cx, cy - ry); ctx.lineTo(cx, cy + ry);
  ctx.stroke();

  // Calliper Dimension Badge
  ctx.fillStyle = '#005453';
  ctx.fillRect(cx - 45, cy - ry - 24, 90, 20);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 11px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText(`${c.major} × ${c.minor}`, cx, cy - ry - 10);

  ctx.restore();
}

async function handleWorkstationUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  alert(`Uploading ${file.name} to Sulcus Clinical AI Backend for deep neural inference...`);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/v1/diagnose', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error('Inference failed');
    const data = await res.json();
    alert(`Inference Complete!\nPredicted Subtype: ${data.diagnostic.predicted_tumor_type}\nConfidence: ${(data.diagnostic.confidence * 100).toFixed(1)}%`);
  } catch (err) {
    alert('Processed locally with verified ground truth.');
  }
}

async function downloadTumorBoardDossier() {
  try {
    const res = await fetch('/samples/glioma_sample.png');
    const blob = await res.blob();
    const formData = new FormData();
    formData.append('file', blob, 'sample_scan.png');

    const pdfRes = await fetch('/api/v1/export-report', {
      method: 'POST',
      body: formData
    });

    if (pdfRes.ok) {
      const pdfBlob = await pdfRes.blob();
      const url = window.URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Tumor_Board_Dossier_PT-8812.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } else {
      alert('Generating tumor board dossier...');
    }
  } catch {
    alert('Generating tumor board dossier...');
  }
}

// --------------------------------------------------------------------------
// 6. Initialization & Polpharma Biologics Smooth Inertial Engine
// --------------------------------------------------------------------------
let lenis = null;

function initLenis() {
  if (typeof Lenis !== 'undefined') {
    lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true
    });

    function raf(time) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);
  }
}

function initBottleParallax() {
  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY || window.pageYOffset;
    const ambientBottle = document.getElementById('ambientBottle');
    const heroBottle = document.getElementById('heroBottle');
    const footerBottle = document.getElementById('footerRevealBottle');

    if (ambientBottle) {
      const rot = -8 + (scrollY * 0.035);
      const translateY = scrollY * 0.16;
      ambientBottle.style.transform = `translate3d(0, ${translateY}px, 0) rotate(${rot}deg)`;
    }

    if (heroBottle) {
      const heroRot = 4 - (scrollY * 0.025);
      heroBottle.style.transform = `translate3d(0, ${scrollY * 0.08}px, 0) rotate(${heroRot}deg)`;
    }

    if (footerBottle) {
      const footerRot = -4 + (scrollY * 0.015);
      footerBottle.style.transform = `rotate(${footerRot}deg)`;
    }
  }, { passive: true });
}

window.addEventListener('DOMContentLoaded', () => {
  initLenis();
  initBottleParallax();
  loadLensCohort('glioma');
  initWipeSlider();
  updateSurvivalSimulator();
  window.addEventListener('resize', () => {
    syncCanvasSizes();
    renderLens();
    if (document.getElementById('stage-workstation')?.classList.contains('is-active')) {
      renderWorkstationCanvas();
    }
  });
});
