"""
Localization Hybride — CLIP + Tesseract OCR
=============================================

Combine :
  1. CLIP (image retrieval via localize_images.py)
  2. Tesseract (OCR classique, robuste, 100% local)

Prérequis :
    1. Installer Tesseract : https://github.com/UB-Mannheim/tesseract/wiki
    2. pip install pytesseract

Usage :
    python localize_hybrid_Tesseract.py ocr-test --image photo.jpg --labels labels.json
    python localize_hybrid_Tesseract.py query --index index.pkl --labels labels.json --image photo.jpg
"""

import os
import json
import argparse
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
# OCR avec Tesseract
# ==============================
class TesseractDetector:
    """Détection de texte via Tesseract (classique, 100% local)."""

    def __init__(self):
        import pytesseract
        # Chemin par défaut de Tesseract sur Windows
        default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default):
            pytesseract.pytesseract.tesseract_cmd = default
        self.pytesseract = pytesseract
        print("[Tesseract] Prêt.")

    def detect(self, image_path, min_confidence=0.3):
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        # Prétraitement : agrandir + contraste
        h, w = img.shape[:2]
        if w < 2000:
            scale = 2000 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Tesseract avec données détaillées (texte + confiance + bbox)
        data = self.pytesseract.image_to_data(
            enhanced, lang="eng+fra", output_type=self.pytesseract.Output.DICT
        )

        detections = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip().upper()
            conf = float(data["conf"][i])

            if conf < 0:  # Tesseract retourne -1 pour les blocs vides
                continue

            conf_norm = conf / 100.0  # Normalise 0-100 → 0-1

            cleaned = ''.join(c for c in text if c.isalnum() or c in '.-_ ').strip()
            if not cleaned:
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

            if conf_norm >= min_confidence:
                detections.append({
                    "text": cleaned,
                    "raw_text": text,
                    "confidence": conf_norm,
                    "bbox": bbox,
                    "source": "tesseract",
                })

        return detections


# ==============================
# Matching labels (identique)
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
# Fusion CLIP + OCR
# ==============================
OCR_MIN_CONFIDENCE = 0.5


def fuse_scores(clip_scores, ocr_matches, clip_weight=0.6, ocr_weight=0.4):
    strong_matches = [m for m in ocr_matches if m["confidence"] >= OCR_MIN_CONFIDENCE]

    if not strong_matches:
        return {room: sc for room, sc in clip_scores.items()}, "clip_only"

    ocr_scores = {}
    for m in strong_matches:
        room = m["room"]
        if room not in ocr_scores or m["confidence"] > ocr_scores[room]:
            ocr_scores[room] = m["confidence"]

    rooms = set(clip_scores.keys()) | set(ocr_scores.keys())
    fused = {}
    for room in rooms:
        cs = clip_scores.get(room, 0.0)
        os_val = ocr_scores.get(room, 0.0)
        if os_val > 0:
            fused[room] = clip_weight * cs + ocr_weight * os_val
        else:
            fused[room] = clip_weight * cs

    return fused, "hybrid"


# ==============================
# Query principale
# ==============================
def hybrid_query(index_path, labels_path, image_path,
                 clip_weight=0.6, ocr_weight=0.4, top_k=10):

    from localize_images import RoomIndex

    idx = RoomIndex.load(index_path)
    clip_result = idx.query(image_path, top_k=top_k)

    detector = TesseractDetector()
    detections = detector.detect(image_path)

    matcher = LabelMatcher(labels_path)
    ocr_matches = matcher.match(detections)

    fused, mode = fuse_scores(clip_result["room_scores"], ocr_matches,
                              clip_weight, ocr_weight)

    best = max(fused, key=fused.get)
    best_sc = fused[best]

    ss = sorted(fused.values(), reverse=True)
    margin = ss[0] - ss[1] if len(ss) > 1 else 1.0

    threshold = clip_result.get("rejection_threshold", 0.3)
    rejected = best_sc < threshold

    return {
        "best_room": best if not rejected else "INCONNU",
        "confidence": best_sc,
        "margin": float(margin),
        "room_scores": fused,
        "fusion_mode": mode,
        "rejected": rejected,
        "clip_scores": clip_result["room_scores"],
        "clip_best": clip_result["best_room"],
        "clip_confidence": clip_result["confidence"],
        "ocr_detections": detections,
        "ocr_matches": ocr_matches,
        "rejection_threshold": threshold,
    }


# ==============================
# Affichage terminal
# ==============================
def _bar(value, max_val, width=25):
    filled = int(value / max(max_val, 1e-9) * width)
    return "█" * filled + "░" * (width - filled)


def cmd_ocr_test(args):
    t0 = time.time()

    detector = TesseractDetector()
    detections = detector.detect(args.image, min_confidence=0.1)

    elapsed = time.time() - t0

    if not detections:
        print(f"Aucun texte détecté ({elapsed:.2f}s)")
        return

    print(f"\n{len(detections)} texte(s) détecté(s) en {elapsed:.2f}s :")
    for d in detections:
        raw = d.get("raw_text", d["text"])
        print(f"  [{d['source']}] raw='{raw}'  clean='{d['text']}'  conf={d['confidence']:.2f}")

    if hasattr(args, 'labels') and args.labels:
        matcher = LabelMatcher(args.labels)
        matched = matcher.match(detections)
        if matched:
            print(f"\n{len(matched)} match(es) :")
            for m in matched:
                print(f"  '{m['text']}' -> {m['room']}  [{m['match_type']}]  conf={m['confidence']:.2f}")
        else:
            print("\nAucun label reconnu.")


def cmd_query(args):
    t0 = time.time()

    r = hybrid_query(
        args.index, args.labels, args.image,
        args.clip_weight, args.ocr_weight,
    )

    elapsed = time.time() - t0
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  LOCALISATION HYBRIDE — CLIP + Tesseract OCR")
    print(f"  Image : {args.image}")
    print(f"{sep}")

    # CLIP
    print(f"\n  [1/3] SCORES CLIP (image retrieval)")
    clip_scores = r["clip_scores"]
    max_clip = max(clip_scores.values()) if clip_scores else 1
    for room, sc in sorted(clip_scores.items(), key=lambda x: -x[1]):
        marker = " ◀ meilleur CLIP" if room == r["clip_best"] else ""
        print(f"    {room:20s}  {_bar(sc, max_clip)}  {sc:.4f}{marker}")

    # OCR
    print(f"\n  [2/3] DÉTECTION TEXTE (Tesseract)")
    detections = r.get("ocr_detections", [])
    if detections:
        print(f"    {len(detections)} texte(s) trouvé(s) :")
        for d in detections:
            print(f"      '{d['text']}'  conf={d['confidence']:.2f}")
    else:
        print("    Aucun texte détecté.")

    ocr_matches = r.get("ocr_matches", [])
    if ocr_matches:
        print(f"\n    {len(ocr_matches)} label(s) reconnu(s) :")
        for m in ocr_matches:
            print(f"      '{m['text']}' -> {m['room']}  [{m['match_type']}]  conf={m['confidence']:.2f}")
    else:
        print("    Aucun label de salle reconnu.")

    # Fusion
    mode = r.get("fusion_mode", "?")
    if mode == "clip_only":
        mode_str = f"CLIP uniquement — OCR ignoré (confiance < {OCR_MIN_CONFIDENCE})"
    else:
        mode_str = f"CLIP {int(args.clip_weight*100)}% + OCR {int(args.ocr_weight*100)}%"
    print(f"\n  [3/3] SCORES FUSIONNÉS ({mode_str})")
    fused = r["room_scores"]
    max_fused = max(fused.values()) if fused else 1
    for room, sc in sorted(fused.items(), key=lambda x: -x[1]):
        clip_sc = clip_scores.get(room, 0.0)
        ocr_sc = next((m["confidence"] for m in ocr_matches if m["room"] == room), None)
        ocr_str = f"  ocr={ocr_sc:.3f}" if ocr_sc is not None else ""
        marker = " ◀" if room == r["best_room"] and not r["rejected"] else ""
        print(f"    {room:20s}  {_bar(sc, max_fused)}  {sc:.4f}  (clip={clip_sc:.3f}{ocr_str}){marker}")

    # Résultat
    print(f"\n{sep}")
    if r["rejected"]:
        print(f"  RÉSULTAT : REJETÉ")
        print(f"  Score {r['confidence']:.4f} < seuil {r.get('rejection_threshold', '?')}")
    else:
        print(f"  RÉSULTAT : {r['best_room']}")
        print(f"  Confiance : {r['confidence']:.4f}   Marge : {r['margin']:.4f}")
    print(f"  Temps    : {elapsed:.2f}s")
    print(f"{sep}\n")


def main():
    p = argparse.ArgumentParser(description="Localisation Hybride — CLIP + Tesseract OCR")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("ocr-test", help="Tester la détection OCR Surya")
    s.add_argument("--image", required=True)
    s.add_argument("--labels", default=None)

    s = sub.add_parser("query", help="Localisation hybride CLIP + Surya")
    s.add_argument("--index", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--clip-weight", type=float, default=0.7)
    s.add_argument("--ocr-weight", type=float, default=0.3)

    args = p.parse_args()
    cmds = {"ocr-test": cmd_ocr_test, "query": cmd_query}
    cmds.get(args.cmd, lambda a: p.print_help())(args)


if __name__ == "__main__":
    main()
