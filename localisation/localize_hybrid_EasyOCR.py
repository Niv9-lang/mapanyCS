"""
Localization Hybride — CLIP + EasyOCR
=====================================

Combine :
  1. CLIP (image retrieval via localize_images.py)
  2. EasyOCR (texte robuste, local)

Deux méthodes de fusion :
  - Linéaire  : Score = α × CLIP + (1-α) × OCR      (α=0.7 par défaut)
  - Softmax   : Normalisation Min-Max + pondération Softmax dynamique (article)

Usage:
    python localize_hybrid_EasyOCR.py query --index index.pkl --labels labels.json --image photo.jpg --method linear
    python localize_hybrid_EasyOCR.py query --index index.pkl --labels labels.json --image photo.jpg --method softmax
    python localize_hybrid_EasyOCR.py ocr-test --image photo.jpg --labels labels.json
"""

import os
import json
import argparse
import base64
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# ==============================
# OCR avec EasyOCR
# ==============================
class TextDetector:
    def __init__(self, languages=None):
        import easyocr
        langs = languages or ['en', 'fr']
        self.reader = easyocr.Reader(langs, gpu=False)

    @staticmethod
    def _prepare_base(image_path):
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            # Fallback PIL pour les formats non supportés par cv2 (HEIC, etc.)
            pil_img = Image.open(str(image_path)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # Agrandit à min 2000px de large
        h, w = img.shape[:2]
        if w < 2000:
            scale = 2000 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def _make_versions(img):
        """Retourne plusieurs traitements colorimétriques d'une même image."""
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        versions = [img]  # original

        # CLAHE
        versions.append(cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR))

        # Inversé + CLAHE (texte clair sur fond sombre)
        inv = cv2.bitwise_not(img)
        gray_inv = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
        versions.append(cv2.cvtColor(clahe.apply(gray_inv), cv2.COLOR_GRAY2BGR))

        # Gamma boost (éclaircit les zones sombres)
        lut = np.array([min(255, int((i / 255) ** 0.4 * 255)) for i in range(256)], dtype=np.uint8)
        versions.append(cv2.LUT(gray.reshape(-1, 1), lut).reshape(gray.shape))

        return versions

    @staticmethod
    def _tiles(img, tile=480, overlap=100):
        """Découpe l'image en tuiles avec chevauchement."""
        import cv2
        h, w = img.shape[:2]
        cuts = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                patch = img[y: y + tile, x: x + tile]
                patch = cv2.resize(patch, (tile * 2, tile * 2), interpolation=cv2.INTER_CUBIC)
                cuts.append(patch)
                x += tile - overlap
            y += tile - overlap
        return cuts

    @staticmethod
    def _extract_sign_crops(img, min_area=300, max_area_ratio=0.15):
        """Détecte les rectangles clairs (panneaux/feuilles sur portes) et les retourne zoomés."""
        import cv2
        h, w = img.shape[:2]
        max_area = h * w * max_area_ratio
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        crops = []
        seen_rects = []  # évite les doublons proches

        for thresh_val in [200, 180, 160, 140]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                x, y, cw, ch = cv2.boundingRect(cnt)
                ratio = cw / max(ch, 1)
                if not (0.3 < ratio < 6.0):
                    continue

                # Évite les doublons (rectangles déjà capturés à un seuil précédent)
                is_dup = False
                for (px, py, pw, ph) in seen_rects:
                    if abs(x - px) < 20 and abs(y - py) < 20:
                        is_dup = True
                        break
                if is_dup:
                    continue
                seen_rects.append((x, y, cw, ch))

                pad = 15
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + ch + pad)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Zoom plus fort pour les petits panneaux
                zoom = max(6, 400 // max(cw, 1))
                zoomed = cv2.resize(crop, (max(cw * zoom, 300), max(ch * zoom, 150)),
                                    interpolation=cv2.INTER_CUBIC)
                crops.append(zoomed)

        return crops

    def detect(self, image_path, min_confidence=0.25):
        import cv2
        base = self._prepare_base(image_path)
        versions = self._make_versions(base)

        raw_all: dict[str, dict] = {}

        def _run_ocr(img_array, label):
            results = self.reader.readtext(
                img_array,
                detail=1,
                paragraph=False,
                contrast_ths=0.05,
                adjust_contrast=0.8,
                text_threshold=0.5,
                low_text=0.3,
                mag_ratio=1.5,
            )
            for (bbox, raw_text, conf) in results:
                key = raw_text.strip().upper()
                cleaned = ''.join(c for c in key if c.isalnum() or c in '.-_ ').strip()
                if not cleaned:
                    continue
                if cleaned not in raw_all or conf > raw_all[cleaned]["confidence"]:
                    raw_all[cleaned] = {
                        "text": cleaned,
                        "raw_text": key,
                        "confidence": float(conf),
                        "bbox": bbox,
                        "source": label,
                    }

        # Passe 1 : image entière — original + inversé CLAHE
        _run_ocr(versions[0], "full_original")
        _run_ocr(versions[2], "full_inv_clahe")

        # Passe 2 : sign crops zoomés
        import cv2
        sign_crops = self._extract_sign_crops(base)
        for crop in sign_crops[:6]:
            _run_ocr(crop, "sign_crop")

        # Passe 3 : tuiles uniquement si rien trouvé
        if not raw_all:
            for tile in self._tiles(versions[0])[:12]:
                _run_ocr(tile, "tile")

        return [d for d in raw_all.values() if d["confidence"] >= min_confidence]


# ==============================
# Fallback : Florence-2 (vision locale)
# ==============================
class FlorenceDetector:
    """Modèle de vision Microsoft Florence-2 — tourne entièrement en local."""

    def __init__(self):
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch

        model_id = "microsoft/Florence-2-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[Florence-2] Chargement du modèle sur {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.dtype = dtype

    def detect(self, image_path, min_confidence=0.0):
        import torch
        from PIL import Image as PILImage

        img = PILImage.open(image_path).convert("RGB")

        task = "<OCR_WITH_REGION>"
        inputs = self.processor(text=task, images=img, return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype) if v.dtype == torch.float32 else v.to(self.device)
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
            )

        raw_output = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            raw_output, task=task, image_size=(img.width, img.height),
        )

        detections = []
        ocr_data = parsed.get("<OCR_WITH_REGION>", {})
        texts = ocr_data.get("labels", [])
        quads = ocr_data.get("quad_boxes", [])

        for i, text in enumerate(texts):
            cleaned = ''.join(c for c in text.strip().upper() if c.isalnum() or c in '.-_ ').strip()
            if not cleaned:
                continue
            bbox = quads[i] if i < len(quads) else []
            detections.append({
                "text": cleaned,
                "raw_text": text.strip().upper(),
                "confidence": 0.85,
                "bbox": bbox,
                "source": "florence2",
            })

        return detections


# ==============================
# Matching labels
# ==============================
class LabelMatcher:
    def __init__(self, labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self.labels: dict[str, str] = {}

        first_val = next(iter(raw.values())) if raw else None
        if isinstance(first_val, list):
            for room, texts in raw.items():
                for t in texts:
                    self.labels[t.strip().upper()] = room
        else:
            for text, room in raw.items():
                self.labels[text.strip().upper()] = room

    def fuzzy_match(self, text):
        import difflib
        if len(text) < 2:
            return None
        cutoff = 0.75 if len(text) <= 3 else 0.65
        matches = difflib.get_close_matches(text, self.labels.keys(), n=1, cutoff=cutoff)
        return matches[0] if matches else None

    def _contains_match(self, text):
        if len(text) < 2:
            return None
        for label in self.labels:
            if len(label) < 2:
                continue
            if label in text and len(label) / len(text) >= 0.6:
                return label
            if text in label and len(text) / len(label) >= 0.6:
                return label
        return None

    def match(self, detections):
        matches = []
        seen = set()

        for det in detections:
            text = det["text"]
            conf = det["confidence"]

            if text in seen or len(text) < 1:
                continue

            if text in self.labels:
                seen.add(text)
                matches.append({
                    "text": text,
                    "room": self.labels[text],
                    "confidence": conf,
                    "match_type": "exact"
                })
                continue

            contained = self._contains_match(text)
            if contained:
                seen.add(text)
                matches.append({
                    "text": text,
                    "room": self.labels[contained],
                    "confidence": conf * 0.88,
                    "match_type": "contains"
                })
                continue

            fuzzy = self.fuzzy_match(text)
            if fuzzy:
                seen.add(text)
                matches.append({
                    "text": text,
                    "room": self.labels[fuzzy],
                    "confidence": conf * 0.70,
                    "match_type": "fuzzy"
                })

        return matches


# ==============================
# Extraction scores OCR par salle
# ==============================
OCR_MIN_CONFIDENCE = 0.5


def _get_ocr_scores(ocr_matches):
    """Extrait le meilleur score OCR par salle (seulement si >= OCR_MIN_CONFIDENCE)."""
    strong = [m for m in ocr_matches if m["confidence"] >= OCR_MIN_CONFIDENCE]
    ocr_scores = {}
    for m in strong:
        room = m["room"]
        if room not in ocr_scores or m["confidence"] > ocr_scores[room]:
            ocr_scores[room] = m["confidence"]
    return ocr_scores


# ======================================================================
# MÉTHODE 1 — Fusion Linéaire (proportionnelle)
#   Score_final = α × S_CLIP + (1-α) × S_OCR
# ======================================================================
def fuse_linear(clip_scores, ocr_matches, clip_weight=0.7, ocr_weight=0.3):
    """
    Fusion linéaire : Score = α × CLIP + (1-α) × OCR.
    Si aucun OCR fiable → 100% CLIP.
    """
    ocr_scores = _get_ocr_scores(ocr_matches)

    if not ocr_scores:
        return {room: sc for room, sc in clip_scores.items()}, "clip_only", {}

    rooms = set(clip_scores.keys()) | set(ocr_scores.keys())
    fused = {}
    details = {}

    for room in rooms:
        cs = clip_scores.get(room, 0.0)
        os_val = ocr_scores.get(room, 0.0)
        if os_val > 0:
            score = clip_weight * cs + ocr_weight * os_val
        else:
            score = clip_weight * cs
        fused[room] = score
        details[room] = {
            "s_clip": cs,
            "s_ocr": os_val,
            "score": score,
        }

    return fused, "linear", details


# ======================================================================
# MÉTHODE 2 — Fusion Softmax (article)
#
#   Étape 1 : S = [S_CLIP, S_OCR] par salle
#   Étape 2 : Normalisation Min-Max par type de score (sur toutes les salles)
#             Ŝ_CLIP = (S_CLIP - min) / (max - min)
#             Ŝ_OCR  = (S_OCR  - min) / (max - min)
#   Étape 3 : Poids dynamiques via Softmax avec température T
#             w = softmax(Ŝ / T)
#   Étape 4 : Score final = Ŝ^T · w  (produit scalaire)
#
#   T contrôle la "dureté" des poids :
#     - T faible (ex: 0.5) → le score dominant prend quasiment tout le poids
#     - T élevé  (ex: 2.0) → poids quasi-égaux entre CLIP et OCR
#     - T = 0.7 (valeur de l'article) → bon compromis
# ======================================================================
def fuse_softmax(clip_scores, ocr_matches, temperature=0.7):
    """
    Fusion par normalisation Min-Max + pondération Softmax dynamique.
    Référence : "Weighted Text-to-Image Retrieval Using CLIP, OCR
    And Keyword Matching" — Équations (2), (3), (4).
    """
    ocr_scores_dict = _get_ocr_scores(ocr_matches)

    if not ocr_scores_dict:
        # Pas d'OCR → CLIP seul, pas de normalisation nécessaire
        return {room: sc for room, sc in clip_scores.items()}, "clip_only", {}

    rooms = sorted(set(clip_scores.keys()) | set(ocr_scores_dict.keys()))

    # ── Étape 1 : vecteurs bruts S_CLIP et S_OCR ──
    clip_vals = np.array([clip_scores.get(r, 0.0) for r in rooms])
    ocr_vals = np.array([ocr_scores_dict.get(r, 0.0) for r in rooms])

    # ── Étape 2 : Normalisation Min-Max par type de score ──
    #   Ramène chaque type de score dans [0, 1] sur l'ensemble des salles
    #   pour les rendre comparables (Équation 2)
    def _minmax(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-9:
            return (arr - mn) / (mx - mn)
        return np.ones_like(arr) * 0.5 if mx > 0 else np.zeros_like(arr)

    clip_norm = _minmax(clip_vals)
    ocr_norm = _minmax(ocr_vals)

    fused = {}
    details = {}

    for i, room in enumerate(rooms):
        # Ŝ = vecteur normalisé pour cette salle
        S_hat = np.array([clip_norm[i], ocr_norm[i]])

        # ── Étape 3 : Softmax avec température T (Équation 3) ──
        #   w_j = exp(Ŝ_j / T) / Σ exp(Ŝ_k / T)
        exp_vals = np.exp(S_hat / temperature)
        w = exp_vals / exp_vals.sum()

        # ── Étape 4 : Score final = Ŝ^T · w (Équation 4) ──
        score = float(np.dot(S_hat, w))

        fused[room] = score
        details[room] = {
            "s_clip_raw": float(clip_vals[i]),
            "s_ocr_raw": float(ocr_vals[i]),
            "s_clip_norm": float(clip_norm[i]),
            "s_ocr_norm": float(ocr_norm[i]),
            "w_clip": float(w[0]),
            "w_ocr": float(w[1]),
            "score": score,
        }

    return fused, "softmax", details


# ==============================
# Query principale
# ==============================
def hybrid_query(index_path, labels_path, image_path,
                 method="linear", clip_weight=0.7, ocr_weight=0.3,
                 temperature=0.7, top_k=10):

    from localize_images import RoomIndex

    idx = RoomIndex.load(index_path)
    clip_result = idx.query(image_path, top_k=top_k)

    detector = TextDetector()
    detections = detector.detect(image_path)

    if not detections:
        try:
            florence = FlorenceDetector()
            detections = florence.detect(image_path)
        except Exception:
            pass

    matcher = LabelMatcher(labels_path)
    ocr_matches = matcher.match(detections)

    # ── Fusion selon la méthode choisie ──
    if method == "softmax":
        fused, mode, fusion_details = fuse_softmax(
            clip_result["room_scores"], ocr_matches, temperature
        )
    else:
        fused, mode, fusion_details = fuse_linear(
            clip_result["room_scores"], ocr_matches, clip_weight, ocr_weight
        )

    best = max(fused, key=fused.get)
    best_sc = fused[best]

    ss = sorted(fused.values(), reverse=True)
    margin = ss[0] - ss[1] if len(ss) > 1 else 1.0

    return {
        "best_room": best,
        "confidence": best_sc,
        "margin": float(margin),
        "room_scores": fused,
        "fusion_mode": mode,
        "fusion_method": method,
        "fusion_details": fusion_details,
        "clip_scores": clip_result["room_scores"],
        "clip_best": clip_result["best_room"],
        "clip_confidence": clip_result["confidence"],
        "ocr_detections": detections,
        "ocr_matches": ocr_matches,
    }


# ==============================
# Commandes CLI
# ==============================
DEFAULT_LABELS = {
    "MG": "zone_MG", "EA": "zone_EA", "EB": "zone_EB",
    "EC": "zone_EC", "ED": "zone_ED", "EE": "zone_EE",
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic', '.heif'}


def cmd_init_ocr(args):
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_LABELS, f, indent=2, ensure_ascii=False)
    print(f"Créé : {args.output}")


def cmd_index_ocr(args):
    """Scanne les images de chaque salle et génère labels.json automatiquement."""
    rooms_dir = Path(args.rooms_dir)
    room_dirs = sorted([
        d for d in rooms_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if not room_dirs:
        print(f"Aucun sous-dossier trouvé dans {rooms_dir}")
        return

    detector = TextDetector()
    min_conf = args.min_confidence

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  INDEXATION OCR — Scan des textes dans {rooms_dir}")
    print(f"{sep}\n")

    labels = {}

    for room_dir in room_dirs:
        room_name = room_dir.name
        images = sorted([
            f for f in room_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not images:
            print(f"  {room_name}: aucune image, skip")
            continue

        max_scan = min(len(images), args.max_images)
        if len(images) > max_scan:
            step = len(images) // max_scan
            images = images[::step][:max_scan]

        print(f"  {room_name}: scan de {len(images)} images...")
        room_texts = set()

        for img_path in images:
            try:
                detections = detector.detect(str(img_path), min_confidence=min_conf)
                for d in detections:
                    text = d["text"]
                    if len(text) >= 2:
                        room_texts.add(text)
            except Exception as e:
                print(f"    Erreur sur {img_path.name}: {e}")

        if room_texts:
            labels[room_name] = sorted(room_texts)
            print(f"    → {len(room_texts)} texte(s) trouvé(s): {sorted(room_texts)}")
        else:
            print(f"    → Aucun texte trouvé")

    output = args.output
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"\n{sep}")
    print(f"  Résultat sauvegardé dans : {output}")
    print(f"  {len(labels)} salle(s) avec du texte détecté")
    print(f"{sep}\n")

    print("  Contenu de labels.json :")
    for room, texts in labels.items():
        print(f"    {room}: {texts}")


def _bar(value, max_val, width=25):
    filled = int(value / max(max_val, 1e-9) * width)
    return "█" * filled + "░" * (width - filled)


def cmd_query(args):
    t0 = time.time()

    r = hybrid_query(
        args.index, args.labels, args.image,
        method=args.method,
        clip_weight=args.clip_weight,
        ocr_weight=args.ocr_weight,
        temperature=args.temperature,
    )

    elapsed = time.time() - t0
    sep = "=" * 70
    method = r["fusion_method"]
    method_label = "LINÉAIRE (α·CLIP + (1-α)·OCR)" if method == "linear" else "SOFTMAX (article)"

    print(f"\n{sep}")
    print(f"  LOCALISATION HYBRIDE — {method_label}")
    print(f"  Image : {args.image}")
    print(f"{sep}")

    clip_scores = r["clip_scores"]
    ocr_matches = r.get("ocr_matches", [])
    detections = r.get("ocr_detections", [])
    details = r.get("fusion_details", {})

    # ══════════════════════════════════════════════════════════════
    #  ÉTAPE 1 : Scores CLIP bruts
    # ══════════════════════════════════════════════════════════════
    print(f"\n  [1/4] SCORES CLIP BRUTS (similarité cosinus)")
    max_clip = max(clip_scores.values()) if clip_scores else 1
    for room, sc in sorted(clip_scores.items(), key=lambda x: -x[1]):
        marker = " ◀ best CLIP" if room == r["clip_best"] else ""
        print(f"    {room:20s}  {_bar(sc, max_clip)}  {sc:.4f}{marker}")
    print(f"\n    → Meilleur CLIP : {r['clip_best']}  (conf={r['clip_confidence']:.4f})")

    # ══════════════════════════════════════════════════════════════
    #  ÉTAPE 2 : Détection OCR
    # ══════════════════════════════════════════════════════════════
    print(f"\n  [2/4] DÉTECTION TEXTE (EasyOCR)")
    if detections:
        print(f"    {len(detections)} texte(s) trouvé(s) :")
        for d in detections:
            src = d.get("source", "?")
            print(f"      [{src}]  '{d['text']}'  conf={d['confidence']:.2f}")
    else:
        print("    Aucun texte détecté dans l'image.")

    if ocr_matches:
        print(f"\n    {len(ocr_matches)} label(s) reconnu(s) :")
        for m in ocr_matches:
            above = "✓" if m["confidence"] >= OCR_MIN_CONFIDENCE else "✗"
            print(f"      '{m['text']}' → {m['room']}  [{m['match_type']}]  "
                  f"conf={m['confidence']:.2f}  {above} seuil={OCR_MIN_CONFIDENCE}")
    else:
        print("    Aucun label de salle reconnu.")

    # ══════════════════════════════════════════════════════════════
    #  ÉTAPE 3 : Fusion (dépend de la méthode)
    # ══════════════════════════════════════════════════════════════
    mode = r.get("fusion_mode", "?")

    if method == "linear":
        _display_linear(args, r, clip_scores, ocr_matches, mode, details)
    else:
        _display_softmax(args, r, clip_scores, ocr_matches, mode, details)

    # ══════════════════════════════════════════════════════════════
    #  ÉTAPE 4 : Résultat final
    # ══════════════════════════════════════════════════════════════
    fused = r["room_scores"]
    print(f"\n{sep}")
    print(f"  RÉSULTAT : {r['best_room']}")
    print(f"  Score    : {r['confidence']:.4f}   Marge : {r['margin']:.4f}")
    print(f"  Méthode  : {method_label}")
    print(f"  Temps    : {elapsed:.2f}s")
    print(f"{sep}\n")


def _display_linear(args, r, clip_scores, ocr_matches, mode, details):
    """Affichage pour la méthode linéaire."""
    if mode == "clip_only":
        mode_str = f"CLIP uniquement — OCR ignoré (confiance < {OCR_MIN_CONFIDENCE})"
    else:
        mode_str = f"Score = {args.clip_weight}×CLIP + {args.ocr_weight}×OCR"

    print(f"\n  [3/4] FUSION LINÉAIRE ({mode_str})")
    print(f"    Formule : Score_final = α × S_CLIP + (1-α) × S_OCR")
    print(f"    α = {args.clip_weight}  (1-α) = {args.ocr_weight}")

    fused = r["room_scores"]
    max_fused = max(fused.values()) if fused else 1

    print()
    for room, sc in sorted(fused.items(), key=lambda x: -x[1]):
        d = details.get(room, {})
        clip_sc = d.get("s_clip", clip_scores.get(room, 0.0))
        ocr_sc = d.get("s_ocr", 0.0)
        ocr_str = f"  ocr={ocr_sc:.3f}" if ocr_sc > 0 else ""
        marker = " ◀" if room == r["best_room"] else ""
        print(f"    {room:20s}  {_bar(sc, max_fused)}  {sc:.4f}  "
              f"(clip={clip_sc:.3f}{ocr_str}){marker}")


def _display_softmax(args, r, clip_scores, ocr_matches, mode, details):
    """Affichage détaillé pour la méthode softmax (article)."""
    T = args.temperature

    if mode == "clip_only":
        print(f"\n  [3/4] FUSION SOFTMAX — OCR ignoré (confiance < {OCR_MIN_CONFIDENCE})")
        print(f"    → Pas de normalisation, scores CLIP bruts utilisés.")
        fused = r["room_scores"]
        max_fused = max(fused.values()) if fused else 1
        for room, sc in sorted(fused.items(), key=lambda x: -x[1]):
            marker = " ◀" if room == r["best_room"] else ""
            print(f"    {room:20s}  {_bar(sc, max_fused)}  {sc:.4f}{marker}")
        return

    print(f"\n  [3/4] FUSION SOFTMAX (article)")
    print(f"    Référence : Weighted Text-to-Image Retrieval Using CLIP, OCR")
    print(f"    Température T = {T}")
    print(f"    (T bas → le score dominant prend plus de poids)")
    print(f"    (T haut → poids quasi-égaux entre CLIP et OCR)")

    # ── 3a : Scores bruts S_CLIP et S_OCR ──
    print(f"\n    ── Étape 3a : Scores bruts S = [S_CLIP, S_OCR] ──")
    for room in sorted(details.keys()):
        d = details[room]
        ocr_str = f"{d['s_ocr_raw']:.4f}" if d['s_ocr_raw'] > 0 else "0"
        print(f"      {room:20s}  S_CLIP={d['s_clip_raw']:.4f}   S_OCR={ocr_str}")

    # ── 3b : Normalisation Min-Max ──
    print(f"\n    ── Étape 3b : Normalisation Min-Max (Éq. 2) ──")
    print(f"      Ŝ = (S − min(S)) / (max(S) − min(S))   [par type de score]")
    for room in sorted(details.keys()):
        d = details[room]
        print(f"      {room:20s}  Ŝ_CLIP={d['s_clip_norm']:.4f}   "
              f"Ŝ_OCR={d['s_ocr_norm']:.4f}")

    # ── 3c : Poids Softmax ──
    print(f"\n    ── Étape 3c : Poids Softmax (Éq. 3) ──")
    print(f"      w = exp(Ŝ/T) / Σ exp(Ŝ/T)   avec T={T}")
    for room in sorted(details.keys()):
        d = details[room]
        print(f"      {room:20s}  w_CLIP={d['w_clip']:.4f}   w_OCR={d['w_ocr']:.4f}")

    # ── 3d : Scores finaux ──
    print(f"\n    ── Étape 3d : Score final Ŝᵀ·w (Éq. 4) ──")
    fused = r["room_scores"]
    max_fused = max(fused.values()) if fused else 1
    for room, sc in sorted(fused.items(), key=lambda x: -x[1]):
        d = details.get(room, {})
        marker = " ◀" if room == r["best_room"] else ""
        w_c = d.get('w_clip', 0)
        w_o = d.get('w_ocr', 0)
        sn_c = d.get('s_clip_norm', 0)
        sn_o = d.get('s_ocr_norm', 0)
        print(f"      {room:20s}  {_bar(sc, max_fused)}  {sc:.4f}  "
              f"({sn_c:.3f}×{w_c:.3f} + {sn_o:.3f}×{w_o:.3f}){marker}")


def cmd_ocr_test(args):
    detector = TextDetector()
    detections = detector.detect(args.image, min_confidence=0.1)

    if not detections:
        print("EasyOCR : rien détecté → fallback Florence-2...")
        try:
            florence = FlorenceDetector()
            detections = florence.detect(args.image)
            if detections:
                print(f"Florence-2 a trouvé {len(detections)} texte(s).")
        except Exception as e:
            print(f"Florence-2 indisponible : {e}")

    if not detections:
        print("Aucun texte détecté (EasyOCR + Florence-2)")
        return

    print(f"\n{len(detections)} texte(s) détecté(s):")
    for d in detections:
        raw = d.get("raw_text", d["text"])
        src = d.get("source", "?")
        print(f"  [{src}] raw='{raw}'  clean='{d['text']}'  conf={d['confidence']:.2f}")

    if hasattr(args, 'labels') and args.labels:
        matcher = LabelMatcher(args.labels)
        matched = matcher.match(detections)
        if matched:
            print(f"\n{len(matched)} match(es):")
            for m in matched:
                print(f"  '{m['text']}' -> {m['room']}  [{m['match_type']}]  conf={m['confidence']:.2f}")
        else:
            print("\nAucun label reconnu.")


def main():
    p = argparse.ArgumentParser(
        description="Localisation Hybride — CLIP + EasyOCR (linéaire ou softmax)"
    )

    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("init-ocr")
    s.add_argument("--output", default="labels.json")

    s = sub.add_parser("index-ocr", help="Scanner les images et générer labels.json")
    s.add_argument("--rooms-dir", required=True)
    s.add_argument("--output", default="labels.json")
    s.add_argument("--max-images", type=int, default=9999)
    s.add_argument("--min-confidence", type=float, default=0.4)

    s = sub.add_parser("query", help="Localisation hybride CLIP + OCR")
    s.add_argument("--index", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--method", choices=["linear", "softmax"], default="linear",
                   help="Méthode de fusion : linear (α·CLIP + (1-α)·OCR) ou softmax (article)")
    s.add_argument("--clip-weight", type=float, default=0.7,
                   help="Poids CLIP (méthode linear seulement, défaut: 0.7)")
    s.add_argument("--ocr-weight", type=float, default=0.3,
                   help="Poids OCR (méthode linear seulement, défaut: 0.3)")
    s.add_argument("--temperature", type=float, default=0.7,
                   help="Température T pour softmax (défaut: 0.7, article)")

    s = sub.add_parser("ocr-test")
    s.add_argument("--image", required=True)
    s.add_argument("--labels", default=None)

    args = p.parse_args()

    cmds = {
        "init-ocr": cmd_init_ocr,
        "index-ocr": cmd_index_ocr,
        "query": cmd_query,
        "ocr-test": cmd_ocr_test
    }

    cmds.get(args.cmd, lambda a: p.print_help())(args)


if __name__ == "__main__":
    main()
