/* ==========================================================================
   ORAL-XAI :: Main Application Script (Alethia 3D Animation + Live Workstation)
   ========================================================================== */

(function () {
    'use strict';

    // --------------------------------------------------------------------------
    // 1. 3D SOLID MEDICAL PHOTO SPHERE ENGINE (Center Spinning Globe)
    // --------------------------------------------------------------------------
    const sphereGlobe = document.getElementById('sphere-globe');
    const sphereScene = document.getElementById('sphere-scene');

    const medicalFacetData = [
        {
            title: "Oral Leukoplakia Macro Scan",
            cat: "Precancerous OPMD",
            acc: "98.7%",
            chip: "LEUKOPLAKIA",
            badge: "OPMD",
            badgeColor: "rgba(0, 255, 157, 0.2)",
            badgeBorder: "#00ff9d",
            src: "assets/optimized/oral_lesion_macro.jpg",
            desc: "Homogenous hyper-keratotic mucosal plaque detected along the lateral tongue margin. Neural classifier indicates 98.7% confidence."
        },
        {
            title: "Specialist Oncologist Review",
            cat: "Clinical Board",
            acc: "99.4%",
            chip: "DR. S. REED, MD",
            badge: "CLINICAL",
            badgeColor: "rgba(0, 243, 255, 0.2)",
            badgeBorder: "#00f3ff",
            src: "assets/optimized/doctor_oncologist.jpg",
            desc: "Board-certified oncologist verifying dual-stream neural activations alongside patient staging history."
        },
        {
            title: "Cancer Histopathology Biopsy",
            cat: "Microscopic Pathology",
            acc: "99.1%",
            chip: "OSCC BIOPSY",
            badge: "BIOPSY",
            badgeColor: "rgba(244, 63, 94, 0.2)",
            badgeBorder: "#f43f5e",
            src: "assets/optimized/cancer_biopsy.jpg",
            desc: "High-resolution H&E stained cellular biopsy slide identifying nuclear pleomorphism, hyperchromasia, and dysplastic keratinization."
        },
        {
            title: "Dual-Stream XAI Grad-CAM",
            cat: "Explainable AI",
            acc: "0.14s",
            chip: "XAI SALIENCY",
            badge: "NEURAL",
            badgeColor: "rgba(139, 92, 246, 0.2)",
            badgeBorder: "#8b5cf6",
            src: "assets/optimized/neural_xai_scan.jpg",
            desc: "Gradient-weighted class activation mapping (Grad-CAM) pinpointing neural saliency heatmaps on dysplastic mucosal margins."
        },
        {
            title: "Patient Care Consultation",
            cat: "Clinical Outcome",
            acc: "Optimal",
            chip: "PATIENT OUTCOME",
            badge: "PROGNOSIS",
            badgeColor: "rgba(0, 243, 255, 0.2)",
            badgeBorder: "#00f3ff",
            src: "assets/optimized/patient_care.jpg",
            desc: "Longitudinal patient outcome telemetry tracking post-biopsy healing, recurrence probabilities, and personalized clinical intervention plans."
        },
        {
            title: "Intraoral Specialist Screening",
            cat: "Intraoral Exam",
            acc: "98.2%",
            chip: "ORAL SCREENING",
            badge: "EXAM",
            badgeColor: "rgba(0, 255, 157, 0.2)",
            badgeBorder: "#00ff9d",
            src: "assets/optimized/oral_lesion_exam.jpg",
            desc: "Direct clinical examination under specialized surgical loupes and mucosal lighting to identify high-risk oral mucosa sites."
        }
    ];

    // Distribute facets symmetrically across latitude bands to form a Solid 3D Sphere
    const SPHERE_RADIUS = 260; // Sphere radius in pixels
    const facets = [];
    const bands = [
        { lat: 0, count: 8, offset: 0 },         // Equator ring
        { lat: 28, count: 6, offset: 15 },      // North temperate
        { lat: -28, count: 6, offset: 45 },     // South temperate
        { lat: 56, count: 3, offset: 0 },       // North polar ring
        { lat: -56, count: 3, offset: 60 }      // South polar ring
    ];

    let assetIdx = 0;
    if (sphereGlobe) {
        bands.forEach(band => {
            const stepTheta = 360 / band.count;
            for (let i = 0; i < band.count; i++) {
                const theta = band.offset + i * stepTheta;
                const phi = band.lat;
                const item = medicalFacetData[assetIdx % medicalFacetData.length];
                assetIdx++;

                const facetEl = document.createElement('div');
                facetEl.className = 'sphere-facet';
                facetEl.setAttribute('data-theta', theta);
                facetEl.setAttribute('data-phi', phi);

                facetEl.innerHTML = `
                    <img src="${item.src}" alt="${item.title}" loading="lazy">
                    <div class="facet-laser"></div>
                    <div class="facet-badge-top" style="background:${item.badgeColor}; border:1px solid ${item.badgeBorder}; color:${item.badgeBorder};">
                        ${item.badge}
                    </div>
                    <div class="facet-chip">${item.chip}</div>
                `;

                // Calculate 3D orientation: points outward radially from the center of the sphere!
                facetEl.style.transform = `rotateY(${theta}deg) rotateX(${-phi}deg) translateZ(${SPHERE_RADIUS}px)`;

                // Hover elevation along radial vector
                facetEl.addEventListener('mouseenter', () => {
                    facetEl.style.transform = `rotateY(${theta}deg) rotateX(${-phi}deg) translateZ(${SPHERE_RADIUS + 38}px) scale(1.15)`;
                });
                facetEl.addEventListener('mouseleave', () => {
                    facetEl.style.transform = `rotateY(${theta}deg) rotateX(${-phi}deg) translateZ(${SPHERE_RADIUS}px) scale(1)`;
                });

                // Click to open clinical modal
                facetEl.addEventListener('click', (e) => {
                    e.stopPropagation();
                    openClinicalModal(item);
                });

                sphereGlobe.appendChild(facetEl);
                facets.push({ el: facetEl, theta, phi, item });
            }
        });
    }

    // Sphere Physics & Continuous Rotation State
    let rotX = -12;
    let rotY = 0;
    let velX = 0;
    let velY = 0.35; // Continuous default spin speed
    let isAutoSpin = true;
    let spinDirection = 1;
    let isHoveringFacet = false;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let lastDragX = 0;
    let lastDragY = 0;

    // Hover slowdown
    if (sphereGlobe) {
        sphereGlobe.addEventListener('mouseenter', () => { isHoveringFacet = true; });
        sphereGlobe.addEventListener('mouseleave', () => { isHoveringFacet = false; });
    }

    // Drag to spin sphere freely in 3D
    if (sphereScene) {
        sphereScene.addEventListener('mousedown', (e) => {
            isDragging = true;
            dragStartX = lastDragX = e.clientX;
            dragStartY = lastDragY = e.clientY;
            velX = 0;
            velY = 0;
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const dx = e.clientX - lastDragX;
            const dy = e.clientY - lastDragY;
            lastDragX = e.clientX;
            lastDragY = e.clientY;

            rotY += dx * 0.45;
            rotX -= dy * 0.4;
            rotX = Math.max(-75, Math.min(75, rotX));

            velY = dx * 0.35;
            velX = -dy * 0.3;
        });

        window.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
            }
        });

        // Touch support
        sphereScene.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                isDragging = true;
                dragStartX = lastDragX = e.touches[0].clientX;
                dragStartY = lastDragY = e.touches[0].clientY;
                velX = 0; velY = 0;
            }
        }, { passive: true });

        window.addEventListener('touchmove', (e) => {
            if (!isDragging || e.touches.length !== 1) return;
            const dx = e.touches[0].clientX - lastDragX;
            const dy = e.touches[0].clientY - lastDragY;
            lastDragX = e.touches[0].clientX;
            lastDragY = e.touches[0].clientY;

            rotY += dx * 0.45;
            rotX -= dy * 0.4;
            rotX = Math.max(-75, Math.min(75, rotX));
            velY = dx * 0.3;
            velX = -dy * 0.25;
        }, { passive: true });

        window.addEventListener('touchend', () => {
            isDragging = false;
        });
    }

    // Sphere Control Buttons
    const btnAuto = document.getElementById('btn-spin-auto');
    const btnReverse = document.getElementById('btn-spin-reverse');
    const btnWire = document.getElementById('btn-sphere-wire');
    const rings = document.querySelectorAll('.orbital-ring');

    if (btnAuto) {
        btnAuto.addEventListener('click', () => {
            isAutoSpin = !isAutoSpin;
            btnAuto.classList.toggle('active', isAutoSpin);
            btnAuto.innerHTML = isAutoSpin ? '<span>⟳ Auto Spin</span>' : '<span>⏸ Paused</span>';
        });
    }

    if (btnReverse) {
        btnReverse.addEventListener('click', () => {
            spinDirection *= -1;
            btnReverse.classList.toggle('active', spinDirection < 0);
        });
    }

    let wireVisible = true;
    if (btnWire) {
        btnWire.addEventListener('click', () => {
            wireVisible = !wireVisible;
            rings.forEach(r => r.style.opacity = wireVisible ? '1' : '0');
            btnWire.classList.toggle('active', wireVisible);
        });
    }

    // Main 60FPS Sphere Render Loop
    function renderSphere() {
        if (!isDragging) {
            if (isAutoSpin) {
                const targetVel = (isHoveringFacet ? 0.08 : 0.4) * spinDirection;
                velY += (targetVel - velY) * 0.05;
                velX += (0 - velX) * 0.05;
            } else {
                velY *= 0.95;
                velX *= 0.95;
            }

            rotY += velY;
            rotX += velX;
            rotX = Math.max(-75, Math.min(75, rotX));
        }

        if (sphereGlobe) {
            sphereGlobe.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg)`;
        }

        requestAnimationFrame(renderSphere);
    }
    requestAnimationFrame(renderSphere);

    // --------------------------------------------------------------------------
    // 2. CLINICAL MODAL TELEMETRY PREVIEW
    // --------------------------------------------------------------------------
    const modal = document.getElementById('modal-overlay');
    const mTitle = document.getElementById('m-title');
    const mCat = document.getElementById('m-cat');
    const mAcc = document.getElementById('m-acc');
    const mDesc = document.getElementById('m-desc');
    const mImg = document.getElementById('m-img');
    const mClose = document.getElementById('m-close');

    function openClinicalModal(data) {
        if (!modal) return;
        mTitle.textContent = data.title;
        mCat.textContent = data.cat;
        mAcc.textContent = data.acc;
        mDesc.textContent = data.desc;
        mImg.src = data.src;
        modal.classList.add('active');
    }

    if (mClose) mClose.addEventListener('click', () => modal.classList.remove('active'));
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.classList.remove('active');
        });
    }
    // --------------------------------------------------------------------------
    // 2. LIVE CLINICAL DIAGNOSTIC WORKSTATION (REAL NEURAL PIPELINE INTEGRATION)
    // --------------------------------------------------------------------------
    const caseSelect = document.getElementById('case-select');
    const imageUpload = document.getElementById('image-upload');
    const clinicalImg = document.getElementById('clinical-img');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const hudName = document.getElementById('hud-name');
    const hudRes = document.getElementById('hud-res');
    const statusTag = document.getElementById('status-tag');

    const rangeOpacity = document.getElementById('range-opacity');
    const valOpacity = document.getElementById('val-opacity');
    const cmapBtns = document.querySelectorAll('.btn-cmap[data-cmap]');
    const checkLime = document.getElementById('check-lime');

    const predClassText = document.getElementById('pred-class-text');
    const predConfText = document.getElementById('pred-conf-text');
    const riskBadge = document.getElementById('risk-badge');
    const confBar = document.getElementById('conf-bar');

    // 4-Class Probability Elements
    const probOsccVal = document.getElementById('prob-oscc-val');
    const probOpmdVal = document.getElementById('prob-opmd-val');
    const probVarVal = document.getElementById('prob-var-val');
    const probNormalVal = document.getElementById('prob-normal-val');

    const barOscc = document.getElementById('bar-oscc');
    const barOpmd = document.getElementById('bar-opmd');
    const barVar = document.getElementById('bar-var');
    const barNormal = document.getElementById('bar-normal');

    const auditText = document.getElementById('audit-text');
    const btnExportJson = document.getElementById('btn-export-json');

    // State Variables
    let currentColormap = 'jet';
    let currentOpacity = 0.55;
    let currentMode = 'gradcam'; // 'gradcam', 'gradcampp', 'lime', 'original'
    let currentAnalysisResult = null;
    let currentFile = null;

    // Connect XAI Modality Mode Buttons
    const xaiModeButtons = document.querySelectorAll('#xai-mode-buttons button');
    function setXAIMode(mode) {
        currentMode = mode;
        xaiModeButtons.forEach(b => {
            if (b.getAttribute('data-mode') === mode) {
                b.classList.add('active');
            } else {
                b.classList.remove('active');
            }
        });
        renderActiveOverlay();
    }

    xaiModeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            setXAIMode(btn.getAttribute('data-mode'));
        });
    });

    // Comparison Card Click Handlers
    const cardCompOrig = document.getElementById('card-comp-original');
    const cardCompGcam = document.getElementById('card-comp-gradcam');
    const cardCompGcampp = document.getElementById('card-comp-gradcampp');
    const cardCompLime = document.getElementById('card-comp-lime');

    if (cardCompOrig) cardCompOrig.addEventListener('click', () => setXAIMode('original'));
    if (cardCompGcam) cardCompGcam.addEventListener('click', () => setXAIMode('gradcam'));
    if (cardCompGcampp) cardCompGcampp.addEventListener('click', () => setXAIMode('gradcampp'));
    if (cardCompLime) cardCompLime.addEventListener('click', () => setXAIMode('lime'));


    // Colormap Selector Buttons
    cmapBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            cmapBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentColormap = btn.getAttribute('data-cmap');
            // If we have an active analysis, re-query server with new colormap
            triggerAnalysis();
        });
    });

    // Opacity Slider
    if (rangeOpacity) {
        rangeOpacity.addEventListener('input', (e) => {
            currentOpacity = e.target.value / 100;
            if (valOpacity) valOpacity.textContent = `${e.target.value}%`;
            // Trigger server re-blend or re-render
            triggerAnalysis(false); // don't re-run LIME for opacity changes
        });
    }

    if (checkLime) {
        checkLime.addEventListener('change', () => {
            triggerAnalysis();
        });
    }

    // Render active overlay from real backend results
    function renderActiveOverlay() {
        if (!currentAnalysisResult || !currentAnalysisResult.overlays) return;
        const overlays = currentAnalysisResult.overlays;

        if (currentMode === 'gradcam' && overlays.gradcam_data_url) {
            clinicalImg.src = overlays.gradcam_data_url;
            if (hudRes) hudRes.textContent = `ATTRIBUTION: GRAD-CAM (1ST-ORDER) • ${currentAnalysisResult.telemetry.cam_latency_ms} MS`;
        } else if (currentMode === 'gradcampp' && overlays.gradcampp_data_url) {
            clinicalImg.src = overlays.gradcampp_data_url;
            if (hudRes) hudRes.textContent = `ATTRIBUTION: GRAD-CAM++ (3RD-ORDER) • ${currentAnalysisResult.telemetry.cam_latency_ms} MS`;
        } else if (currentMode === 'lime' && overlays.lime_data_url) {
            clinicalImg.src = overlays.lime_data_url;
            if (hudRes) hudRes.textContent = `ATTRIBUTION: LIME SUPERPIXELS (${currentAnalysisResult.telemetry.lime_samples} SAMPLES) • ${currentAnalysisResult.telemetry.lime_latency_ms} MS`;
        } else if (overlays.original_data_url) {
            clinicalImg.src = overlays.original_data_url;
            if (hudRes) hudRes.textContent = `VIEW: ORIGINAL CLINICAL PHOTO (PREPROCESSED 224x224)`;
        }
    }

    // Update workstation UI elements with real 4-class predictions and probabilities
    function updateWorkstationUI(res) {
        currentAnalysisResult = res;

        if (hudName) {
            hudName.textContent = (res.filename || 'INPUT_SCAN.JPG').toUpperCase();
        }

        if (predClassText) {
            predClassText.textContent = res.predicted_class;
        }

        const confScore = res.confidence_percentage !== undefined ? res.confidence_percentage : (res.confidence <= 1.0 ? res.confidence * 100 : res.confidence);
        if (predConfText) {
            predConfText.textContent = `${confScore.toFixed(1)}%`;
        }

        if (confBar) {
            confBar.style.width = `${Math.min(100, Math.max(0, confScore))}%`;
        }

        if (riskBadge) {
            riskBadge.textContent = res.risk_tier;
            riskBadge.className = `risk-badge ${res.risk_class}`;
        }

        const p = res.probabilities || {};
        const toPct = (val) => {
            const v = val || 0;
            return v <= 1.0 ? v * 100 : v;
        };

        const pOscc = toPct(p['Oral Cancer']);
        const pOpmd = toPct(p['OPMD']);
        const pVar = toPct(p['Variation from normal']);
        const pNorm = toPct(p['Normal']);

        if (probOsccVal) probOsccVal.textContent = `${pOscc.toFixed(1)}%`;
        if (probOpmdVal) probOpmdVal.textContent = `${pOpmd.toFixed(1)}%`;
        if (probVarVal) probVarVal.textContent = `${pVar.toFixed(1)}%`;
        if (probNormalVal) probNormalVal.textContent = `${pNorm.toFixed(1)}%`;

        if (barOscc) barOscc.style.width = `${pOscc}%`;
        if (barOpmd) barOpmd.style.width = `${pOpmd}%`;
        if (barVar) barVar.style.width = `${pVar}%`;
        if (barNormal) barNormal.style.width = `${pNorm}%`;

        // Update Sequential Multi-Modal Attribution Comparison Pipeline Cards
        const compOrig = document.getElementById('comp-img-original');
        const compPredClass = document.getElementById('comp-pred-class');
        const compPredConf = document.getElementById('comp-pred-conf');
        const compPredBadge = document.getElementById('comp-pred-badge');
        const compGcam = document.getElementById('comp-img-gradcam');
        const compGcampp = document.getElementById('comp-img-gradcampp');
        const compLime = document.getElementById('comp-img-lime');

        if (res.overlays) {
            if (compOrig && res.overlays.original_data_url) compOrig.src = res.overlays.original_data_url;
            if (compGcam && (res.overlays.gradcam_overlay_data_url || res.overlays.gradcam_data_url)) {
                compGcam.src = res.overlays.gradcam_overlay_data_url || res.overlays.gradcam_data_url;
            }
            if (compGcampp && (res.overlays.gradcampp_overlay_data_url || res.overlays.gradcampp_data_url)) {
                compGcampp.src = res.overlays.gradcampp_overlay_data_url || res.overlays.gradcampp_data_url;
            }
            if (compLime && (res.overlays.lime_overlay_data_url || res.overlays.lime_data_url)) {
                compLime.src = res.overlays.lime_overlay_data_url || res.overlays.lime_data_url;
            }
        }

        if (compPredClass) compPredClass.textContent = res.predicted_class;
        if (compPredConf) compPredConf.textContent = `${confScore.toFixed(1)}%`;
        if (compPredBadge) {
            compPredBadge.textContent = res.risk_tier;
            compPredBadge.className = `risk-badge ${res.risk_class}`;
        }

        if (auditText && res.clinical_audit) {
            auditText.innerHTML = res.clinical_audit.map(item => `• ${item}`).join('<br>');
        }

        if (statusTag) {
            statusTag.textContent = `${res.model_name.toUpperCase()} ACTIVE • ${res.telemetry.total_latency_ms} MS`;
        }

        renderActiveOverlay();
    }


    // Execute Real Server Analysis
    async function triggerAnalysis(includeLime = true) {
        try {
            if (statusTag) statusTag.textContent = 'NEURAL INFERENCE COMPUTING...';

            const formData = new FormData();
            formData.append('colormap', currentColormap);
            formData.append('opacity', currentOpacity);
            formData.append('lime', (checkLime && checkLime.checked && includeLime) ? 'true' : 'false');
            formData.append('lime_samples', '25');

            if (currentFile) {
                formData.append('file', currentFile);
            } else {
                const presetVal = caseSelect ? caseSelect.value : 'leukoplakia';
                formData.append('preset', presetVal);
            }

            const res = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const errJson = await res.json().catch(() => ({}));
                throw new Error(errJson.error || `HTTP ${res.status}`);
            }

            const data = await res.json();
            updateWorkstationUI(data);
        } catch (err) {
            console.error('[oral-XAI] Analysis Error:', err);
            if (statusTag) statusTag.textContent = 'INFERENCE ERROR';
            if (auditText) {
                auditText.innerHTML = `⚠️ <span style="color: #f43f5e;">Analysis Error: ${err.message}</span><br>Ensure server is running with best_model.keras.`;
            }
        }
    }

    // Preset Case Dropdown Listener
    if (caseSelect) {
        caseSelect.addEventListener('change', () => {
            currentFile = null; // Clear custom upload when preset is explicitly chosen
            triggerAnalysis();
        });
    }

    // Custom File Upload Listener
    if (imageUpload) {
        imageUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                currentFile = file;
                triggerAnalysis();
            }
        });
    }

    // Export Clinical JSON Diagnostic Report
    if (btnExportJson) {
        btnExportJson.addEventListener('click', () => {
            if (!currentAnalysisResult) {
                alert('Please run a diagnostic analysis first.');
                return;
            }

            const report = {
                platform: 'oral-XAI :: Explainable Deep Learning Diagnostic Workstation',
                model: currentAnalysisResult.model_name,
                modelCheckpoint: currentAnalysisResult.model_path,
                timestamp: new Date().toISOString(),
                inputImage: currentAnalysisResult.filename,
                prediction: {
                    predictedClass: currentAnalysisResult.predicted_class,
                    classIndex: currentAnalysisResult.predicted_class_index,
                    confidenceScore: `${currentAnalysisResult.confidence}%`,
                    riskTier: currentAnalysisResult.risk_tier,
                    multiclassProbabilityDistribution: currentAnalysisResult.probabilities,
                    rawProbabilityVector: currentAnalysisResult.raw_probabilities,
                    probabilitySum: currentAnalysisResult.probability_sum
                },
                explainabilityAttributions: {
                    activeModalityView: currentMode,
                    gradcamColormap: currentColormap,
                    blendingOpacity: `${Math.round(currentOpacity * 100)}%`,
                    limeSuperpixelSamples: currentAnalysisResult.telemetry.lime_samples,
                    attributionType: 'Model attention / attribution visualization'
                },
                telemetry: currentAnalysisResult.telemetry,
                clinicalReasoningAudit: currentAnalysisResult.clinical_audit,
                regulatoryDisclaimer: currentAnalysisResult.disclaimer
            };

            const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(report, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute('href', dataStr);
            downloadAnchor.setAttribute('download', `ORAL_XAI_REPORT_${Date.now()}.json`);
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        });
    }

    // Initial Workstation Load
    setTimeout(() => {
        triggerAnalysis();
    }, 200);

})();
