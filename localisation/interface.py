"""
Interface web locale pour la localisation hybride CLIP + OCR.
Lance un serveur Flask sur http://localhost:5000

Usage :
    python interface.py --index index.pkl --labels labels.json

Dépendance :
    pip install flask
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

# Ajoute le dossier courant au path pour les imports locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Globaux chargés au démarrage
INDEX_PATH = None
LABELS_PATH = None
DETECTOR = None
MATCHER = None
ROOM_INDEX = None


def init_models():
    global DETECTOR, MATCHER, ROOM_INDEX
    from localize_hybrid_EasyOCR import TextDetector, LabelMatcher
    from localize_images import RoomIndex

    print("[Interface] Chargement de l'index CLIP...")
    ROOM_INDEX = RoomIndex.load(INDEX_PATH)
    print("[Interface] Chargement du détecteur OCR (EasyOCR)...")
    DETECTOR = TextDetector()
    print("[Interface] Chargement des labels...")
    MATCHER = LabelMatcher(LABELS_PATH)
    print("[Interface] Prêt !")


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Localisation — CLIP + OCR</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
        min-height: 100vh;
    }

    .header {
        text-align: center;
        padding: 30px 20px 10px;
    }
    .header h1 {
        font-size: 2em;
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header p { color: #888; margin-top: 5px; font-size: 0.95em; }

    .container {
        max-width: 1200px;
        margin: 20px auto;
        padding: 0 20px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
    }

    .panel {
        background: rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
    }

    .panel h2 {
        font-size: 1.1em;
        margin-bottom: 15px;
        color: #4facfe;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Zone de drop ── */
    .drop-zone {
        border: 2px dashed rgba(79,172,254,0.4);
        border-radius: 12px;
        padding: 40px 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        background: rgba(79,172,254,0.03);
        min-height: 250px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .drop-zone:hover, .drop-zone.dragover {
        border-color: #4facfe;
        background: rgba(79,172,254,0.1);
    }
    .drop-zone img {
        max-width: 100%;
        max-height: 350px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .drop-zone .icon { font-size: 3em; margin-bottom: 10px; }
    .drop-zone p { color: #888; }

    .btn-row { display: flex; gap: 10px; margin-top: 15px; }

    .btn {
        flex: 1;
        padding: 12px 20px;
        border: none;
        border-radius: 10px;
        font-size: 1em;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-linear {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .btn-linear:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(102,126,234,0.4); }
    .btn-softmax {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
    }
    .btn-softmax:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(245,87,108,0.4); }

    /* ── Loader ── */
    .loader {
        display: none;
        text-align: center;
        padding: 30px;
    }
    .loader.active { display: block; }
    .spinner {
        width: 40px; height: 40px;
        border: 4px solid rgba(255,255,255,0.1);
        border-top-color: #4facfe;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin: 0 auto 15px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Résultat ── */
    .result-card {
        background: rgba(79,172,254,0.08);
        border: 1px solid rgba(79,172,254,0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        text-align: center;
    }
    .result-room {
        font-size: 2.2em;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .result-meta {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 10px;
        color: #aaa;
        font-size: 0.9em;
    }
    .result-meta span { font-weight: 600; color: #4facfe; }

    .method-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
        margin-top: 8px;
    }
    .badge-linear { background: rgba(102,126,234,0.3); color: #a8b8ff; }
    .badge-softmax { background: rgba(245,87,108,0.3); color: #ffb0be; }

    /* ── Barres de score ── */
    .score-list { list-style: none; }
    .score-item {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
        gap: 10px;
    }
    .score-label {
        width: 100px;
        font-size: 0.85em;
        font-weight: 600;
        text-align: right;
        flex-shrink: 0;
    }
    .score-bar-bg {
        flex: 1;
        height: 22px;
        background: rgba(255,255,255,0.06);
        border-radius: 6px;
        overflow: hidden;
        position: relative;
    }
    .score-bar {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s ease;
        min-width: 2px;
    }
    .score-bar.clip { background: linear-gradient(90deg, #3a7bd5, #4facfe); }
    .score-bar.fused-linear { background: linear-gradient(90deg, #667eea, #764ba2); }
    .score-bar.fused-softmax { background: linear-gradient(90deg, #f093fb, #f5576c); }
    .score-bar.best { box-shadow: 0 0 10px rgba(79,172,254,0.5); }
    .score-val {
        width: 55px;
        font-size: 0.8em;
        color: #aaa;
        flex-shrink: 0;
    }
    .score-item.is-best .score-label { color: #4facfe; }
    .score-item.is-best .score-val { color: #4facfe; font-weight: 700; }

    /* ── OCR ── */
    .ocr-tag {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 5px 12px;
        margin: 3px;
        font-size: 0.85em;
    }
    .ocr-tag .conf { color: #888; font-size: 0.8em; margin-left: 5px; }
    .ocr-match {
        background: rgba(46,204,113,0.15);
        border-color: rgba(46,204,113,0.3);
    }
    .ocr-match .arrow { color: #2ecc71; margin: 0 5px; }

    /* ── Softmax details ── */
    .detail-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8em;
        margin-top: 10px;
    }
    .detail-table th {
        background: rgba(255,255,255,0.05);
        padding: 6px 8px;
        text-align: center;
        color: #888;
        font-weight: 600;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .detail-table td {
        padding: 5px 8px;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .detail-table tr.best-row td { color: #4facfe; font-weight: 600; }

    .section-title {
        font-size: 0.9em;
        color: #888;
        margin: 15px 0 8px;
        padding-bottom: 5px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    .time-info {
        text-align: center;
        color: #666;
        font-size: 0.85em;
        margin-top: 10px;
    }

    .full-width { grid-column: 1 / -1; }

    @media (max-width: 768px) {
        .container { grid-template-columns: 1fr; }
    }
</style>
</head>
<body>

<div class="header">
    <h1>Localisation CLIP + OCR</h1>
    <p>Déposez une photo pour identifier la salle</p>
</div>

<div class="container">
    <!-- ── Panneau gauche : Image ── -->
    <div class="panel">
        <h2>📷 Image</h2>
        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <div class="icon">📁</div>
            <p>Glissez une image ici<br>ou cliquez pour choisir</p>
        </div>
        <input type="file" id="fileInput" accept="image/*" style="display:none">

        <div class="btn-row">
            <button class="btn btn-linear" id="btnLinear" disabled onclick="analyze('linear')">
                Méthode Linéaire
            </button>
            <button class="btn btn-softmax" id="btnSoftmax" disabled onclick="analyze('softmax')">
                Méthode Softmax
            </button>
        </div>
    </div>

    <!-- ── Panneau droit : Résultats ── -->
    <div class="panel">
        <h2>📊 Résultats</h2>
        <div id="loader" class="loader">
            <div class="spinner"></div>
            <p>Analyse en cours...</p>
        </div>
        <div id="results"></div>
    </div>

    <!-- ── Panneau pleine largeur : Détails ── -->
    <div class="panel full-width" id="detailsPanel" style="display:none">
        <h2>🔍 Détails</h2>
        <div id="details"></div>
    </div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const btnLinear = document.getElementById('btnLinear');
const btnSoftmax = document.getElementById('btnSoftmax');
let currentFile = null;

// ── Drag & Drop ──
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) loadFile(fileInput.files[0]); });

function loadFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        dropZone.innerHTML = '<img src="' + e.target.result + '">';
    };
    reader.readAsDataURL(file);
    btnLinear.disabled = false;
    btnSoftmax.disabled = false;
    document.getElementById('results').innerHTML = '';
    document.getElementById('detailsPanel').style.display = 'none';
}

// ── Analyse ──
async function analyze(method) {
    if (!currentFile) return;

    btnLinear.disabled = true;
    btnSoftmax.disabled = true;
    document.getElementById('loader').classList.add('active');
    document.getElementById('results').innerHTML = '';
    document.getElementById('detailsPanel').style.display = 'none';

    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('method', method);

    try {
        const resp = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await resp.json();
        displayResults(data, method);
    } catch (err) {
        document.getElementById('results').innerHTML = '<p style="color:#e74c3c">Erreur : ' + err.message + '</p>';
    }

    document.getElementById('loader').classList.remove('active');
    btnLinear.disabled = false;
    btnSoftmax.disabled = false;
}

function displayResults(data, method) {
    const r = document.getElementById('results');
    const badgeClass = method === 'linear' ? 'badge-linear' : 'badge-softmax';
    const badgeText = method === 'linear' ? 'Linéaire (α=0.7)' : 'Softmax (T=0.7)';
    const barClass = method === 'linear' ? 'fused-linear' : 'fused-softmax';

    // ── Résultat principal ──
    let html = '<div class="result-card">';
    html += '<div class="result-room">' + data.best_room + '</div>';
    html += '<div class="result-meta">';
    html += '<div>Score <span>' + data.confidence.toFixed(4) + '</span></div>';
    html += '<div>Marge <span>' + data.margin.toFixed(4) + '</span></div>';
    html += '<div>Temps <span>' + data.elapsed.toFixed(2) + 's</span></div>';
    html += '</div>';
    html += '<div class="method-badge ' + badgeClass + '">' + badgeText + '</div>';
    html += '</div>';

    // ── Scores fusionnés ──
    html += '<div class="section-title">Scores fusionnés</div>';
    html += '<ul class="score-list">';
    const sorted = Object.entries(data.room_scores).sort((a,b) => b[1] - a[1]);
    const maxScore = sorted[0][1];
    for (const [room, sc] of sorted) {
        const pct = (sc / maxScore * 100).toFixed(1);
        const isBest = room === data.best_room;
        html += '<li class="score-item ' + (isBest ? 'is-best' : '') + '">';
        html += '<div class="score-label">' + room + '</div>';
        html += '<div class="score-bar-bg"><div class="score-bar ' + barClass + (isBest ? ' best' : '') + '" style="width:' + pct + '%"></div></div>';
        html += '<div class="score-val">' + sc.toFixed(4) + '</div>';
        html += '</li>';
    }
    html += '</ul>';

    // ── Scores CLIP ──
    html += '<div class="section-title">Scores CLIP (bruts)</div>';
    html += '<ul class="score-list">';
    const clipSorted = Object.entries(data.clip_scores).sort((a,b) => b[1] - a[1]);
    const maxClip = clipSorted[0][1];
    for (const [room, sc] of clipSorted) {
        const pct = (sc / maxClip * 100).toFixed(1);
        const isBest = room === data.clip_best;
        html += '<li class="score-item ' + (isBest ? 'is-best' : '') + '">';
        html += '<div class="score-label">' + room + '</div>';
        html += '<div class="score-bar-bg"><div class="score-bar clip" style="width:' + pct + '%"></div></div>';
        html += '<div class="score-val">' + sc.toFixed(4) + '</div>';
        html += '</li>';
    }
    html += '</ul>';

    r.innerHTML = html;

    // ── Détails (panneau bas) ──
    const d = document.getElementById('details');
    const dp = document.getElementById('detailsPanel');
    dp.style.display = 'block';
    let dhtml = '';

    // OCR détections
    dhtml += '<div class="section-title">Textes détectés (OCR)</div>';
    if (data.ocr_detections.length === 0) {
        dhtml += '<p style="color:#888">Aucun texte détecté sur l\'image.</p>';
    } else {
        for (const det of data.ocr_detections) {
            dhtml += '<span class="ocr-tag">' + det.text + '<span class="conf">' + det.confidence.toFixed(2) + '</span></span>';
        }
    }

    // OCR matches
    dhtml += '<div class="section-title" style="margin-top:15px">Labels reconnus</div>';
    if (data.ocr_matches.length === 0) {
        dhtml += '<p style="color:#888">Aucun label de salle reconnu.</p>';
    } else {
        for (const m of data.ocr_matches) {
            dhtml += '<span class="ocr-tag ocr-match">' + m.text + '<span class="arrow">→</span>' + m.room;
            dhtml += ' <span class="conf">[' + m.match_type + '] ' + m.confidence.toFixed(2) + '</span></span>';
        }
    }

    // Détails softmax
    if (method === 'softmax' && data.fusion_details && Object.keys(data.fusion_details).length > 0) {
        dhtml += '<div class="section-title" style="margin-top:15px">Détails Softmax (normalisation + poids)</div>';
        dhtml += '<table class="detail-table"><tr>';
        dhtml += '<th>Salle</th><th>S_CLIP</th><th>S_OCR</th><th>Ŝ_CLIP</th><th>Ŝ_OCR</th><th>w_CLIP</th><th>w_OCR</th><th>Score final</th>';
        dhtml += '</tr>';
        const detailSorted = Object.entries(data.fusion_details).sort((a,b) => b[1].score - a[1].score);
        for (const [room, det] of detailSorted) {
            const isBest = room === data.best_room;
            dhtml += '<tr class="' + (isBest ? 'best-row' : '') + '">';
            dhtml += '<td>' + room + '</td>';
            dhtml += '<td>' + det.s_clip_raw.toFixed(4) + '</td>';
            dhtml += '<td>' + det.s_ocr_raw.toFixed(4) + '</td>';
            dhtml += '<td>' + det.s_clip_norm.toFixed(4) + '</td>';
            dhtml += '<td>' + det.s_ocr_norm.toFixed(4) + '</td>';
            dhtml += '<td>' + det.w_clip.toFixed(4) + '</td>';
            dhtml += '<td>' + det.w_ocr.toFixed(4) + '</td>';
            dhtml += '<td>' + det.score.toFixed(4) + '</td>';
            dhtml += '</tr>';
        }
        dhtml += '</table>';
    }

    d.innerHTML = dhtml;
}
</script>

</body>
</html>
"""


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/analyze', methods=['POST'])
def analyze():
    from localize_hybrid_EasyOCR import (
        hybrid_query, fuse_linear, fuse_softmax,
        _get_ocr_scores, OCR_MIN_CONFIDENCE
    )

    if 'image' not in request.files:
        return jsonify({"error": "Pas d'image"}), 400

    file = request.files['image']
    method = request.form.get('method', 'linear')

    # Sauvegarde temporaire
    suffix = Path(file.filename).suffix or '.png'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    tmp.close()

    try:
        t0 = time.time()

        # CLIP
        clip_result = ROOM_INDEX.query(tmp.name, top_k=10)

        # OCR
        detections = DETECTOR.detect(tmp.name)
        ocr_matches = MATCHER.match(detections)

        # Fusion
        if method == "softmax":
            fused, mode, details = fuse_softmax(
                clip_result["room_scores"], ocr_matches, temperature=0.7
            )
        else:
            fused, mode, details = fuse_linear(
                clip_result["room_scores"], ocr_matches, clip_weight=0.7, ocr_weight=0.3
            )

        best = max(fused, key=fused.get)
        best_sc = fused[best]
        ss = sorted(fused.values(), reverse=True)
        margin = ss[0] - ss[1] if len(ss) > 1 else 1.0

        elapsed = time.time() - t0

        result = {
            "best_room": best,
            "confidence": best_sc,
            "margin": margin,
            "room_scores": fused,
            "fusion_mode": mode,
            "clip_scores": clip_result["room_scores"],
            "clip_best": clip_result["best_room"],
            "clip_confidence": clip_result["confidence"],
            "ocr_detections": [
                {"text": d["text"], "confidence": d["confidence"], "source": d.get("source", "?")}
                for d in detections
            ],
            "ocr_matches": [
                {"text": m["text"], "room": m["room"], "match_type": m["match_type"], "confidence": m["confidence"]}
                for m in ocr_matches
            ],
            "fusion_details": details,
            "elapsed": elapsed,
        }

        return jsonify(result)

    finally:
        os.unlink(tmp.name)


def main():
    global INDEX_PATH, LABELS_PATH

    parser = argparse.ArgumentParser(description="Interface web — Localisation CLIP + OCR")
    parser.add_argument("--index", required=True, help="Fichier index.pkl")
    parser.add_argument("--labels", required=True, help="Fichier labels.json")
    parser.add_argument("--port", type=int, default=5000, help="Port (défaut: 5000)")
    args = parser.parse_args()

    INDEX_PATH = args.index
    LABELS_PATH = args.labels

    init_models()

    print(f"\n{'='*50}")
    print(f"  Interface disponible sur : http://localhost:{args.port}")
    print(f"{'='*50}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
