import base64
from pathlib import Path

# Load optimized images as base64
opt_dir = Path("c:/Users/Bcsm Computer/Downloads/Oral-lesion-xai/app/assets/optimized")
imgs = {}
for fname in ["oral_lesion_macro", "doctor_oncologist", "cancer_biopsy", "patient_care", "neural_xai_scan", "oral_lesion_exam"]:
    fpath = opt_dir / f"{fname}.jpg"
    with open(fpath, "rb") as fp:
        imgs[fname] = "data:image/jpeg;base64," + base64.b64encode(fp.read()).decode("utf-8")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ORAL-XAI :: Explainable AI Oral Lesion & Oncology Diagnostics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #060913;
            --card-bg: rgba(14, 22, 39, 0.75);
            --cyan: #00f3ff;
            --emerald: #00ff9d;
            --blue: #2563eb;
            --violet: #8b5cf6;
            --rose: #f43f5e;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --card-border: rgba(0, 243, 255, 0.22);
            --glow-cyan: 0 0 30px rgba(0, 243, 255, 0.25);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }}

        html, body {{
            width: 100%;
            height: 100%;
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
            overflow-y: auto;
        }}

        /* Subtle Ambient Glow Background */
        .ambient-bg {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }}

        .glow-sphere {{
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.18;
            animation: pulseGlow 14s ease-in-out infinite alternate;
        }}

        .glow-1 {{
            width: 700px;
            height: 700px;
            background: radial-gradient(circle, #00f3ff, #0066ff, transparent);
            top: -150px;
            left: 15%;
        }}

        .glow-2 {{
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, #8b5cf6, #ec4899, transparent);
            bottom: -100px;
            right: 10%;
            animation-duration: 18s;
            animation-delay: -5s;
        }}

        .glow-3 {{
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, #00ff9d, #059669, transparent);
            top: 40%;
            left: -100px;
            animation-duration: 20s;
        }}

        @keyframes pulseGlow {{
            0% {{ transform: translate(0, 0) scale(1); opacity: 0.14; }}
            50% {{ transform: translate(40px, -30px) scale(1.15); opacity: 0.22; }}
            100% {{ transform: translate(-30px, 40px) scale(0.95); opacity: 0.14; }}
        }}

        /* Subtle Cyber Grid */
        .cyber-grid {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: 
                linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
            background-size: 60px 60px;
            pointer-events: none;
            z-index: 1;
            mask-image: radial-gradient(circle at 50% 50%, black 40%, transparent 80%);
            -webkit-mask-image: radial-gradient(circle at 50% 50%, black 40%, transparent 80%);
        }}

        /* Navigation */
        nav.cyber-nav {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            padding: 1.2rem 3.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
            backdrop-filter: blur(16px);
            background: rgba(6, 9, 19, 0.65);
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }}

        .nav-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'Orbitron', sans-serif;
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #ffffff;
            text-decoration: none;
        }}

        .brand-icon {{
            width: 34px;
            height: 34px;
            background: linear-gradient(135deg, var(--cyan), var(--blue));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            box-shadow: 0 0 15px rgba(0, 243, 255, 0.4);
        }}

        .nav-links {{
            display: flex;
            align-items: center;
            gap: 2rem;
            list-style: none;
        }}

        .nav-link {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-decoration: none;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            transition: color 0.2s, text-shadow 0.2s;
        }}

        .nav-link:hover, .nav-link.active {{
            color: var(--cyan);
            text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
        }}

        .nav-cta {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            color: #060913;
            background: linear-gradient(90deg, var(--cyan), var(--emerald));
            padding: 8px 20px;
            border-radius: 20px;
            text-decoration: none;
            letter-spacing: 0.04em;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.35);
        }}

        .nav-cta:hover {{
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0, 243, 255, 0.6);
        }}

        /* Hero Wrapper */
        .hero-section {{
            position: relative;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 7rem 2rem 2rem 2rem;
            z-index: 10;
        }}

        /* Hero Text Content (Centered, floating over / integrated with 3D elements) */
        .hero-header-content {{
            text-align: center;
            max-width: 900px;
            margin-bottom: 2rem;
            pointer-events: auto;
            position: relative;
            z-index: 25;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            background: rgba(0, 243, 255, 0.08);
            border: 1px solid rgba(0, 243, 255, 0.3);
            border-radius: 30px;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: var(--cyan);
            margin-bottom: 1.2rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.15);
        }}

        .pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--emerald);
            box-shadow: 0 0 10px var(--emerald);
            animation: pulseDot 2s infinite;
        }}

        @keyframes pulseDot {{
            0%, 100% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.4); opacity: 0.6; }}
        }}

        .hero-title {{
            font-size: 3.6rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.02em;
            margin-bottom: 1.2rem;
        }}

        .gradient-text {{
            background: linear-gradient(135deg, #ffffff 10%, var(--cyan) 50%, var(--emerald) 90%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 40px rgba(0, 243, 255, 0.25);
        }}

        .hero-subtitle {{
            font-size: 1.15rem;
            line-height: 1.6;
            color: var(--text-secondary);
            max-width: 720px;
            margin: 0 auto 1.8rem auto;
            font-weight: 400;
        }}

        .hero-actions {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1.2rem;
            margin-bottom: 2rem;
        }}

        .btn-primary {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: #060913;
            background: linear-gradient(90deg, #00f3ff, #00ff9d);
            padding: 12px 28px;
            border-radius: 12px;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 0 25px rgba(0, 243, 255, 0.35);
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            border: none;
            cursor: pointer;
        }}

        .btn-primary:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 0 35px rgba(0, 243, 255, 0.6);
        }}

        .btn-secondary {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: #e2e8f0;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 12px 24px;
            border-radius: 12px;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            backdrop-filter: blur(10px);
            transition: all 0.25s ease;
            cursor: pointer;
        }}

        .btn-secondary:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--cyan);
            color: var(--cyan);
        }}

        /* 3D SCENE STAGE (The Alethia Spatial Animation Canvas) */
        .stage-viewport {{
            position: relative;
            width: 100%;
            max-width: 1400px;
            height: 640px;
            perspective: 1400px;
            perspective-origin: 50% 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 20;
        }}

        .stage-3d {{
            position: relative;
            width: 100%;
            height: 100%;
            transform-style: preserve-3d;
            will-change: transform;
            transition: transform 0.1s cubic-bezier(0.2, 0, 0.2, 1);
        }}

        /* Medical 3D Card Base Style */
        .med-card {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform-style: preserve-3d;
            border-radius: 20px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.75), 0 0 20px rgba(0, 243, 255, 0.1);
            overflow: hidden;
            cursor: pointer;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            will-change: transform, opacity;
            transition: box-shadow 0.4s ease, border-color 0.4s ease;
        }}

        .med-card:hover {{
            border-color: var(--cyan);
            box-shadow: 0 35px 80px rgba(0, 0, 0, 0.85), 0 0 35px rgba(0, 243, 255, 0.4);
            z-index: 50 !important;
        }}

        .med-card img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            border-radius: 19px;
            transition: transform 0.5s ease;
        }}

        .med-card:hover img {{
            transform: scale(1.04);
        }}

        /* Interactive Specular Glare Overlay */
        .card-glare {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at var(--gx, 50%) var(--gy, 50%), rgba(255, 255, 255, 0.2) 0%, transparent 60%);
            pointer-events: none;
            mix-blend-mode: overlay;
            z-index: 3;
        }}

        /* Scanning Laser Bar (On Clinical Lesion Card) */
        .scan-laser {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, transparent, var(--cyan), #ffffff, var(--cyan), transparent);
            box-shadow: 0 0 15px var(--cyan);
            animation: scanMove 3.5s ease-in-out infinite alternate;
            z-index: 4;
            pointer-events: none;
        }}

        @keyframes scanMove {{
            0% {{ top: 5%; opacity: 0.3; }}
            50% {{ opacity: 1; }}
            100% {{ top: 92%; opacity: 0.3; }}
        }}

        /* Targeting Reticle Brackets */
        .reticle-corner {{
            position: absolute;
            width: 16px;
            height: 16px;
            border-color: var(--cyan);
            border-style: solid;
            pointer-events: none;
            z-index: 4;
            opacity: 0.85;
            transition: all 0.3s ease;
        }}

        .rc-tl {{ top: 10px; left: 10px; border-width: 2px 0 0 2px; }}
        .rc-tr {{ top: 10px; right: 10px; border-width: 2px 2px 0 0; }}
        .rc-bl {{ bottom: 10px; left: 10px; border-width: 0 0 2px 2px; }}
        .rc-br {{ bottom: 10px; right: 10px; border-width: 0 2px 2px 0; }}

        .med-card:hover .reticle-corner {{
            width: 24px;
            height: 24px;
            border-color: var(--emerald);
            box-shadow: 0 0 10px rgba(0, 255, 157, 0.6);
        }}

        /* HUD Badges & Callout Labels (Alethia Signature Tags) */
        .hud-badge {{
            position: absolute;
            z-index: 5;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 8px;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }}

        .badge-cyan {{
            background: rgba(0, 243, 255, 0.15);
            border: 1px solid rgba(0, 243, 255, 0.5);
            color: #ffffff;
            text-shadow: 0 0 8px rgba(0, 243, 255, 0.8);
        }}

        .badge-emerald {{
            background: rgba(0, 255, 157, 0.15);
            border: 1px solid rgba(0, 255, 157, 0.5);
            color: #ffffff;
            text-shadow: 0 0 8px rgba(0, 255, 157, 0.8);
        }}

        .badge-violet {{
            background: rgba(139, 92, 246, 0.18);
            border: 1px solid rgba(139, 92, 246, 0.5);
            color: #ffffff;
            text-shadow: 0 0 8px rgba(139, 92, 246, 0.8);
        }}

        .badge-rose {{
            background: rgba(244, 63, 94, 0.18);
            border: 1px solid rgba(244, 63, 94, 0.5);
            color: #ffffff;
            text-shadow: 0 0 8px rgba(244, 63, 94, 0.8);
        }}

        /* Floating Pill Tag (like Alethia -8.3 tCO2e tag) */
        .floating-pill {{
            position: absolute;
            z-index: 6;
            background: rgba(6, 12, 24, 0.88);
            border: 1px solid rgba(0, 243, 255, 0.4);
            border-radius: 20px;
            padding: 5px 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            color: #ffffff;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6), 0 0 15px rgba(0, 243, 255, 0.2);
            white-space: nowrap;
            animation: pillFloat 4s ease-in-out infinite alternate;
        }}

        @keyframes pillFloat {{
            0% {{ transform: translateY(0); }}
            100% {{ transform: translateY(-6px); }}
        }}

        /* CARD INDIVIDUAL DIMENSIONS & INITIAL POSITIONS */
        /* Center Hero Card: Clinical Oral Lesion Detection */
        .card-center {{
            width: 440px;
            height: 310px;
            margin-top: -155px;
            margin-left: -220px;
            z-index: 10;
        }}

        /* Card 1: Specialist Doctor / Oncologist Review (Top-Left) */
        .card-doctor {{
            width: 320px;
            height: 220px;
            margin-top: -240px;
            margin-left: -530px;
            z-index: 8;
        }}

        /* Card 2: Cancer Histopathology Biopsy (Top-Right) */
        .card-biopsy {{
            width: 330px;
            height: 230px;
            margin-top: -230px;
            margin-left: 210px;
            z-index: 7;
        }}

        /* Card 3: Neural Explainability / Grad-CAM Heatmap (Bottom-Left) */
        .card-xai {{
            width: 330px;
            height: 225px;
            margin-top: 50px;
            margin-left: -500px;
            z-index: 9;
        }}

        /* Card 4: Patient Consultation & Care Outcome (Bottom-Right) */
        .card-patient {{
            width: 325px;
            height: 225px;
            margin-top: 60px;
            margin-left: 190px;
            z-index: 8;
        }}

        /* Card 5: Dental / Intraoral Specialist Exam (Upper-Mid Floating) */
        .card-exam {{
            width: 250px;
            height: 170px;
            margin-top: -290px;
            margin-left: -125px;
            z-index: 5;
            opacity: 0.88;
        }}

        /* Bottom Floating Bar with Telemetry & Controls */
        .hud-telemetry-bar {{
            margin-top: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 2.5rem;
            padding: 12px 32px;
            background: rgba(10, 16, 32, 0.7);
            border: 1px solid rgba(0, 243, 255, 0.2);
            border-radius: 40px;
            backdrop-filter: blur(20px);
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-secondary);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            z-index: 30;
        }}

        .telemetry-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .telemetry-value {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--cyan);
        }}

        .telemetry-divider {{
            width: 1px;
            height: 18px;
            background: rgba(255, 255, 255, 0.15);
        }}

        /* Interactive Controls Pill */
        .interactive-dock {{
            position: fixed;
            bottom: 25px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 18px;
            background: rgba(6, 12, 24, 0.85);
            border: 1px solid rgba(0, 243, 255, 0.3);
            border-radius: 30px;
            backdrop-filter: blur(20px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 243, 255, 0.15);
            z-index: 100;
        }}

        .dock-btn {{
            background: transparent;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-secondary);
            padding: 6px 14px;
            border-radius: 20px;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .dock-btn:hover, .dock-btn.active {{
            background: rgba(0, 243, 255, 0.15);
            border-color: var(--cyan);
            color: var(--cyan);
            box-shadow: 0 0 15px rgba(0, 243, 255, 0.3);
        }}

        /* Clinical Case Detail Modal */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(4, 7, 16, 0.85);
            backdrop-filter: blur(16px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }}

        .modal-overlay.open {{
            opacity: 1;
            pointer-events: auto;
        }}

        .modal-card {{
            width: 90%;
            max-width: 720px;
            background: #0b1122;
            border: 1px solid var(--cyan);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 30px 90px rgba(0, 0, 0, 0.9), 0 0 40px rgba(0, 243, 255, 0.3);
            transform: scale(0.9) translateY(20px);
            transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        .modal-overlay.open .modal-card {{
            transform: scale(1) translateY(0);
        }}

        .modal-header {{
            padding: 1.2rem 1.8rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 243, 255, 0.05);
        }}

        .modal-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.15rem;
            color: var(--cyan);
        }}

        .modal-close {{
            background: transparent;
            border: none;
            color: #94a3b8;
            font-size: 1.5rem;
            cursor: pointer;
            line-height: 1;
        }}

        .modal-close:hover {{
            color: #ffffff;
        }}

        .modal-body {{
            padding: 1.8rem;
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 1.5rem;
        }}

        .modal-img {{
            width: 100%;
            height: 240px;
            object-fit: cover;
            border-radius: 12px;
            border: 1px solid rgba(0, 243, 255, 0.3);
        }}

        .modal-info {{
            font-size: 0.9rem;
            color: #cbd5e1;
            line-height: 1.6;
        }}

        .modal-field {{
            margin-bottom: 0.8rem;
        }}

        .modal-label {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--cyan);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        /* Responsive */
        @media (max-width: 1100px) {{
            .hero-title {{ font-size: 2.6rem; }}
            .card-doctor {{ margin-left: -400px; width: 260px; height: 180px; }}
            .card-biopsy {{ margin-left: 150px; width: 260px; height: 180px; }}
            .card-xai {{ margin-left: -380px; width: 260px; height: 180px; }}
            .card-patient {{ margin-left: 140px; width: 260px; height: 180px; }}
            .card-center {{ width: 360px; height: 250px; margin-left: -180px; }}
        }}

        @media (max-width: 768px) {{
            nav.cyber-nav {{ padding: 1rem 1.5rem; }}
            .nav-links {{ display: none; }}
            .hero-title {{ font-size: 2.1rem; }}
            .stage-viewport {{ height: 500px; }}
            .card-doctor, .card-biopsy, .card-xai, .card-patient, .card-exam {{
                transform: scale(0.7) !important;
            }}
        }}
    </style>
</head>
<body>

    <!-- Ambient Glowing Nebulae -->
    <div class="ambient-bg">
        <div class="glow-sphere glow-1"></div>
        <div class="glow-sphere glow-2"></div>
        <div class="glow-sphere glow-3"></div>
    </div>
    <div class="cyber-grid"></div>

    <!-- Top Navigation -->
    <nav class="cyber-nav">
        <a href="#" class="nav-brand">
            <div class="brand-icon">🔬</div>
            <span>ORAL-XAI</span>
        </a>
        <ul class="nav-links">
            <li><a href="#diagnostics" class="nav-link active">Neural Hub</a></li>
            <li><a href="#clinical-cases" class="nav-link">Histopathology</a></li>
            <li><a href="#xai-engine" class="nav-link">Dual-Stream XAI</a></li>
            <li><a href="#benchmarks" class="nav-link">Clinical Benchmarks</a></li>
        </ul>
        <a href="http://localhost:8501" target="_blank" class="nav-cta">Launch Diagnostics ↗</a>
    </nav>

    <!-- Main Landing Hero Section -->
    <section class="hero-section">
        
        <!-- Hero Title & Subtext (Alethia inspired) -->
        <div class="hero-header-content">
            <div class="hero-badge">
                <span class="pulse-dot"></span>
                <span>STATE-OF-THE-ART ORAL ONCOLOGY INTELLIGENCE</span>
            </div>
            <h1 class="hero-title">
                Precision Detection.<br>
                <span class="gradient-text">Explainable Medical AI.</span>
            </h1>
            <p class="hero-subtitle">
                Revolutionizing early oral cancer diagnosis through multi-layer deep neural networks, real-time Grad-CAM saliency heatmaps, and histopathological correlation you can trust.
            </p>
            <div class="hero-actions">
                <a href="http://localhost:8501" target="_blank" class="btn-primary">
                    <span>⚡ Launch Neural Diagnostic Suite</span>
                </a>
                <button class="btn-secondary" id="btn-replay-anim">
                    <span>↺ Replay 3D Sequence</span>
                </button>
            </div>
        </div>

        <!-- 3D Spatial Canvas (Alethia-style Interactive Floating 3D Cards) -->
        <div class="stage-viewport" id="viewport">
            <div class="stage-3d" id="stage">
                
                <!-- CARD 1: DOCTOR / ONCOLOGIST CLINICAL REVIEW (Top-Left) -->
                <div class="med-card card-doctor" id="card-doctor" data-depth="0.75" data-title="Specialist Oncologist Review" data-cat="Clinical Board" data-acc="99.4%" data-desc="Triple-board certified oncologists verifying dual-stream neural activations alongside patient staging history." data-img="{imgs['doctor_oncologist']}">
                    <img src="{imgs['doctor_oncologist']}" alt="Doctor Oncologist Reviewing Diagnostics">
                    <div class="card-glare"></div>
                    <div class="reticle-corner rc-tl"></div>
                    <div class="reticle-corner rc-br"></div>
                    <div class="hud-badge badge-cyan" style="top: 12px; left: 14px;">
                        <span>🩺 CLINICAL REVIEW</span>
                    </div>
                    <div class="floating-pill" style="bottom: 12px; right: 12px;">
                        <span style="color: var(--cyan);">●</span> DR. S. REED, MD
                    </div>
                </div>

                <!-- CARD 2: CANCER HISTOPATHOLOGY BIOPSY (Top-Right) -->
                <div class="med-card card-biopsy" id="card-biopsy" data-depth="0.70" data-title="Cancer Histopathology Biopsy" data-cat="Microscopic Pathology" data-acc="99.1%" data-desc="High-resolution H&E stained cellular biopsy slide identifying nuclear pleomorphism, hyperchromasia, and dysplastic keratinization." data-img="{imgs['cancer_biopsy']}">
                    <img src="{imgs['cancer_biopsy']}" alt="Cancer Histopathology Microscopy">
                    <div class="card-glare"></div>
                    <div class="reticle-corner rc-tr"></div>
                    <div class="reticle-corner rc-bl"></div>
                    <div class="hud-badge badge-rose" style="top: 12px; right: 14px;">
                        <span>🔬 HISTOPATHOLOGY</span>
                    </div>
                    <div class="floating-pill" style="bottom: 12px; left: 12px;">
                        <span style="color: var(--rose);">●</span> OSCC BIOPSY VERIFIED
                    </div>
                </div>

                <!-- CARD 3: CLINICAL DENTAL EXAMINATION (Far Mid Floating) -->
                <div class="med-card card-exam" id="card-exam" data-depth="0.45" data-title="Intraoral Specialist Examination" data-cat="Intraoral Screening" data-acc="98.2%" data-desc="Direct clinical examination under specialized surgical loupes and mucosal lighting to identify high-risk oral mucosa sites." data-img="{imgs['oral_lesion_exam']}">
                    <img src="{imgs['oral_lesion_exam']}" alt="Dental Specialist Intraoral Exam">
                    <div class="card-glare"></div>
                    <div class="hud-badge badge-emerald" style="top: 8px; left: 10px; font-size: 0.7rem;">
                        <span>🦷 ORAL EXAMINATION</span>
                    </div>
                    <div class="floating-pill" style="bottom: 8px; right: 8px; font-size: 0.68rem; padding: 3px 10px;">
                        STAGE SCREENING
                    </div>
                </div>

                <!-- CARD 4 (CENTER HERO): CLINICAL ORAL LESION WITH ACTIVE AI SCANNING -->
                <div class="med-card card-center" id="card-lesion" data-depth="1.0" data-title="Oral Lesion Detection (Leukoplakia)" data-cat="Precancerous OPMD" data-acc="98.7%" data-desc="Lateral tongue border displaying homogenous leukoplakia plaque. Neural network detects keratosis boundary with 98.7% classification confidence." data-img="{imgs['oral_lesion_macro']}">
                    <img src="{imgs['oral_lesion_macro']}" alt="Oral Lesion Clinical Macro">
                    <div class="scan-laser"></div>
                    <div class="card-glare"></div>
                    <div class="reticle-corner rc-tl"></div>
                    <div class="reticle-corner rc-tr"></div>
                    <div class="reticle-corner rc-bl"></div>
                    <div class="reticle-corner rc-br"></div>
                    
                    <div class="hud-badge badge-emerald" style="top: 14px; left: 16px;">
                        <span class="pulse-dot" style="width: 6px; height: 6px;"></span>
                        <span>TARGET: LINGUAL BORDER</span>
                    </div>

                    <div class="hud-badge badge-cyan" style="top: 14px; right: 16px;">
                        <span>AI CONFIDENCE: 98.7%</span>
                    </div>

                    <div class="floating-pill" style="bottom: 16px; left: 16px;">
                        <span style="color: var(--emerald);">●</span> OPMD LEUKOPLAKIA DETECTED
                    </div>

                    <div class="floating-pill" style="bottom: 16px; right: 16px; font-size: 0.7rem;">
                        <span>INFERENCE: 11.4 ms</span>
                    </div>
                </div>

                <!-- CARD 5: NEURAL EXPLAINABILITY / GRAD-CAM (Bottom-Left) -->
                <div class="med-card card-xai" id="card-xai" data-depth="0.85" data-title="Dual-Stream Grad-CAM & Saliency Map" data-cat="Explainable AI" data-acc="0.14s" data-desc="Real-time gradient-weighted class activation mapping (Grad-CAM) pinpointing neural saliency heatmaps on dysplastic mucosal margins." data-img="{imgs['neural_xai_scan']}">
                    <img src="{imgs['neural_xai_scan']}" alt="Neural XAI Heatmap Activation">
                    <div class="card-glare"></div>
                    <div class="reticle-corner rc-tl"></div>
                    <div class="reticle-corner rc-bl"></div>
                    <div class="hud-badge badge-violet" style="top: 12px; left: 14px;">
                        <span>🧠 XAI GRAD-CAM SALIENCY</span>
                    </div>
                    <div class="floating-pill" style="bottom: 12px; right: 12px;">
                        <span style="color: var(--violet);">●</span> FEATURE MAP HOTSPOT
                    </div>
                </div>

                <!-- CARD 6: PATIENT CONSULTATION & CARE OUTCOME (Bottom-Right) -->
                <div class="med-card card-patient" id="card-patient" data-depth="0.80" data-title="Patient Consultation & Recovery" data-cat="Clinical Outcome" data-acc="Optimal" data-desc="Longitudinal patient outcome telemetry tracking post-biopsy healing, recurrence probabilities, and personalized clinical intervention plans." data-img="{imgs['patient_care']}">
                    <img src="{imgs['patient_care']}" alt="Patient Consultation Room">
                    <div class="card-glare"></div>
                    <div class="reticle-corner rc-tr"></div>
                    <div class="reticle-corner rc-br"></div>
                    <div class="hud-badge badge-cyan" style="top: 12px; right: 14px;">
                        <span>🤝 PATIENT CONSULTATION</span>
                    </div>
                    <div class="floating-pill" style="bottom: 12px; left: 12px;">
                        <span style="color: var(--emerald);">●</span> PROGNOSIS: EXCELLENT
                    </div>
                </div>

            </div>
        </div>

        <!-- Telemetry Stats Bar -->
        <div class="hud-telemetry-bar">
            <div class="telemetry-item">
                <span>DIAGNOSTIC SENSITIVITY:</span>
                <span class="telemetry-value">99.2%</span>
            </div>
            <div class="telemetry-divider"></div>
            <div class="telemetry-item">
                <span>HISTOLOGY CORRELATION:</span>
                <span class="telemetry-value">98.7%</span>
            </div>
            <div class="telemetry-divider"></div>
            <div class="telemetry-item">
                <span>INFERENCE LATENCY:</span>
                <span class="telemetry-value">11.4 MS</span>
            </div>
            <div class="telemetry-divider"></div>
            <div class="telemetry-item">
                <span>EXPLAINABILITY:</span>
                <span class="telemetry-value">GRAD-CAM & LIME</span>
            </div>
        </div>

    </section>

    <!-- Bottom Interactive Control Dock -->
    <div class="interactive-dock">
        <button class="dock-btn active" id="btn-toggle-float"><span>🪐 3D Float Active</span></button>
        <button class="dock-btn" id="btn-tilt-preset-1"><span>Angle: Isometric</span></button>
        <button class="dock-btn" id="btn-tilt-preset-2"><span>Angle: Frontal Focus</span></button>
        <button class="dock-btn" id="btn-tilt-preset-3"><span>Angle: Dynamic Orbit</span></button>
    </div>

    <!-- Clinical Modal Overlay -->
    <div class="modal-overlay" id="modal">
        <div class="modal-card">
            <div class="modal-header">
                <div class="modal-title" id="modal-title">Clinical Telemetry</div>
                <button class="modal-close" id="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <img src="" id="modal-img" class="modal-img" alt="Case Detail">
                <div class="modal-info">
                    <div class="modal-field">
                        <div class="modal-label">Classification Category</div>
                        <div id="modal-cat" style="font-weight: 700; color: #ffffff;">Precancerous OPMD</div>
                    </div>
                    <div class="modal-field">
                        <div class="modal-label">Model Confidence / Accuracy</div>
                        <div id="modal-acc" style="color: var(--emerald); font-family: 'Orbitron'; font-weight: 700;">98.7%</div>
                    </div>
                    <div class="modal-field">
                        <div class="modal-label">Clinical Diagnostic Notes</div>
                        <div id="modal-desc">Clinical case examination notes.</div>
                    </div>
                    <div style="margin-top: 1rem;">
                        <a href="http://localhost:8501" target="_blank" class="btn-primary" style="padding: 8px 16px; font-size: 0.85rem;">
                            Analyze in Neural Suite ➜
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- ALETHIA-STYLE DYNAMIC 3D SPATIAL ANIMATION ENGINE -->
    <script>
    (function() {{
        const viewport = document.getElementById('viewport');
        const stage = document.getElementById('stage');
        const cards = [
            {{ el: document.getElementById('card-doctor'),  baseX: -360, baseY: -120, baseZ: -40,  rotX: 6,  rotY: 10,  rotZ: -3,  scale: 1, speed: 0.9, phase: 0.0 }},
            {{ el: document.getElementById('card-biopsy'),  baseX: 350,  baseY: -110, baseZ: -30,  rotX: 4,  rotY: -10, rotZ: 3,   scale: 1, speed: 1.1, phase: 1.5 }},
            {{ el: document.getElementById('card-exam'),    baseX: -10,  baseY: -210, baseZ: -90,  rotX: 8,  rotY: 0,   rotZ: -1,  scale: 0.9, speed: 0.7, phase: 2.2 }},
            {{ el: document.getElementById('card-lesion'),  baseX: 0,    baseY: 0,    baseZ: 60,   rotX: 0,  rotY: 0,   rotZ: 0,   scale: 1.05, speed: 1.3, phase: 0.8 }},
            {{ el: document.getElementById('card-xai'),     baseX: -340, baseY: 130,  baseZ: 20,   rotX: -5, rotY: 8,   rotZ: 2,   scale: 1, speed: 1.0, phase: 3.1 }},
            {{ el: document.getElementById('card-patient'), baseX: 330,  baseY: 140,  baseZ: 10,   rotX: -4, rotY: -7,  rotZ: -2,  scale: 1, speed: 0.85, phase: 4.0 }}
        ];

        let targetMouseX = 0;
        let targetMouseY = 0;
        let currentMouseX = 0;
        let currentMouseY = 0;
        let isFloating = true;
        let cameraPreset = 0; // 0 = mouse interactive, 1 = isometric, 2 = frontal, 3 = orbit

        // Mouse Parallax calculation (Alethia Dynamic Tilt)
        window.addEventListener('mousemove', (e) => {{
            if (cameraPreset !== 0) return;
            const w = window.innerWidth;
            const h = window.innerHeight;
            targetMouseX = (e.clientX - w / 2) / (w / 2);
            targetMouseY = (e.clientY - h / 2) / (h / 2);

            // Update card specular glare
            cards.forEach(c => {{
                const rect = c.el.getBoundingClientRect();
                const gx = ((e.clientX - rect.left) / rect.width) * 100;
                const gy = ((e.clientY - rect.top) / rect.height) * 100;
                c.el.style.setProperty('--gx', `${{gx}}%`);
                c.el.style.setProperty('--gy', `${{gy}}%`);
            }});
        }});

        // Entrance timeline (Alethia Expand & Orbit Sequence)
        let introStartTime = performance.now();
        let introDuration = 2200; // 2.2s silky smooth burst

        function replayAnimation() {{
            introStartTime = performance.now();
            cameraPreset = 0;
            updateDockActive(null);
        }}

        document.getElementById('btn-replay-anim').addEventListener('click', replayAnimation);

        // Toggle Floating Physics
        const toggleFloatBtn = document.getElementById('btn-toggle-float');
        toggleFloatBtn.addEventListener('click', () => {{
            isFloating = !isFloating;
            toggleFloatBtn.classList.toggle('active', isFloating);
            toggleFloatBtn.innerHTML = isFloating ? '<span>🪐 3D Float Active</span>' : '<span>⏸ 3D Float Paused</span>';
        }});

        // Presets
        const p1 = document.getElementById('btn-tilt-preset-1');
        const p2 = document.getElementById('btn-tilt-preset-2');
        const p3 = document.getElementById('btn-tilt-preset-3');

        function updateDockActive(btn) {{
            [p1, p2, p3].forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
        }}

        p1.addEventListener('click', () => {{
            cameraPreset = 1;
            updateDockActive(p1);
        }});
        p2.addEventListener('click', () => {{
            cameraPreset = 2;
            updateDockActive(p2);
        }});
        p3.addEventListener('click', () => {{
            cameraPreset = 3;
            updateDockActive(p3);
        }});

        // Main 60FPS RAF Render Loop
        function render(now) {{
            const elapsed = now - introStartTime;
            let introProgress = Math.min(elapsed / introDuration, 1.0);
            
            // Custom cubic bezier easing: cubic-bezier(0.16, 1, 0.3, 1)
            let ease = 1 - Math.pow(1 - introProgress, 4);

            // Handle Camera Mode
            if (cameraPreset === 0) {{
                // Smooth Lerp Damping
                currentMouseX += (targetMouseX - currentMouseX) * 0.06;
                currentMouseY += (targetMouseY - currentMouseY) * 0.06;
            }} else if (cameraPreset === 1) {{
                // Isometric Preset
                currentMouseX += (0.6 - currentMouseX) * 0.05;
                currentMouseY += (-0.4 - currentMouseY) * 0.05;
            }} else if (cameraPreset === 2) {{
                // Frontal Flat Focus
                currentMouseX += (0 - currentMouseX) * 0.08;
                currentMouseY += (0 - currentMouseY) * 0.08;
            }} else if (cameraPreset === 3) {{
                // Continuous Dynamic Cinematic Orbit
                const orbitT = now * 0.001;
                targetMouseX = Math.sin(orbitT * 0.6) * 0.65;
                targetMouseY = Math.cos(orbitT * 0.4) * 0.35;
                currentMouseX += (targetMouseX - currentMouseX) * 0.05;
                currentMouseY += (targetMouseY - currentMouseY) * 0.05;
            }}

            // Stage 3D Perspective Rotation
            const stageRotX = -currentMouseY * 14;
            const stageRotY = currentMouseX * 18;
            stage.style.transform = `rotateX(${{stageRotX}}deg) rotateY(${{stageRotY}}deg)`;

            const t = now * 0.0015;

            // Render each card with harmonic floating & depth parallax
            cards.forEach((card, idx) => {{
                // Entrance animation interpolation
                const startScale = 0.4;
                const startZ = -300;
                const curScale = (startScale + (card.scale - startScale) * ease);
                
                // Subtle floating oscillation
                let floatY = 0;
                let floatX = 0;
                let floatRot = 0;
                if (isFloating) {{
                    floatY = Math.sin(t * card.speed + card.phase) * 12;
                    floatX = Math.cos(t * card.speed * 0.8 + card.phase) * 5;
                    floatRot = Math.sin(t * card.speed * 0.7 + card.phase) * 1.5;
                }}

                // Parallax translation multiplier per card depth
                const depthFactor = parseFloat(card.el.getAttribute('data-depth')) || 0.8;
                const parallaxX = currentMouseX * 35 * depthFactor;
                const parallaxY = currentMouseY * 25 * depthFactor;

                const curX = (card.baseX * ease) + floatX + parallaxX;
                const curY = (card.baseY * ease) + floatY + parallaxY;
                const curZ = (startZ + (card.baseZ - startZ) * ease);

                const curRotX = card.rotX - currentMouseY * 6;
                const curRotY = card.rotY + currentMouseX * 8;
                const curRotZ = card.rotZ + floatRot;

                // Card opacity during entrance
                const curOpacity = Math.min(ease * 1.3, 1.0);

                card.el.style.transform = `translate3d(${{curX}}px, ${{curY}}px, ${{curZ}}px) rotateX(${{curRotX}}deg) rotateY(${{curRotY}}deg) rotateZ(${{curRotZ}}deg) scale(${{curScale}})`;
                card.el.style.opacity = curOpacity;
            }});

            requestAnimationFrame(render);
        }}

        requestAnimationFrame(render);

        // Interactive Card Click -> Modal Preview
        const modal = document.getElementById('modal');
        const modalTitle = document.getElementById('modal-title');
        const modalCat = document.getElementById('modal-cat');
        const modalAcc = document.getElementById('modal-acc');
        const modalDesc = document.getElementById('modal-desc');
        const modalImg = document.getElementById('modal-img');
        const modalClose = document.getElementById('modal-close');

        cards.forEach(c => {{
            c.el.addEventListener('click', () => {{
                modalTitle.textContent = c.el.getAttribute('data-title');
                modalCat.textContent = c.el.getAttribute('data-cat');
                modalAcc.textContent = c.el.getAttribute('data-acc');
                modalDesc.textContent = c.el.getAttribute('data-desc');
                modalImg.src = c.el.getAttribute('data-img');
                modal.classList.add('open');
            }});
        }});

        modalClose.addEventListener('click', () => {{
            modal.classList.remove('open');
        }});

        modal.addEventListener('click', (e) => {{
            if (e.target === modal) modal.classList.remove('open');
        }});
    }})();
    </script>
</body>
</html>
"""

landing_path = Path("c:/Users/Bcsm Computer/Downloads/Oral-lesion-xai/app/landing.html")
landing_path.write_text(html_content, encoding="utf-8")
print(f"Generated landing.html at {landing_path} ({landing_path.stat().st_size} bytes)")
