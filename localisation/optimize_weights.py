"""
Optimisation des poids CLIP / OCR
==================================
Deux modes :

1. Image unique (exploration visuelle) :
    python optimize_weights.py --index index.pkl --labels labels.json --image photo.jpg

2. Jeu de test (justification statistique) :
    python optimize_weights.py --index index.pkl --labels labels.json --test-dir ./test_images/

    Structure attendue de test_images/ :
        test_images/
            salle1/
                photo1.jpg
                photo2.jpg
            salle3/
                photo3.jpg
                ...
    Le nom du sous-dossier = la vraie salle (doit correspondre aux clés de labels.json).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


OCR_MIN_CONFIDENCE = 0.5


def get_raw_scores(index_path, labels_path, image_path):
    """
    Récupère les scores CLIP et OCR bruts (une seule fois).
    Retourne (clip_scores, ocr_scores, ocr_matches, detections).
    """
    from localize_images import RoomIndex
    from localize_hybrid import TextDetector, FlorenceDetector, LabelMatcher

    print("  Chargement de l'index CLIP...")
    idx = RoomIndex.load(index_path)
    clip_result = idx.query(image_path, top_k=10)

    print("  Détection OCR...")
    detector = TextDetector()
    detections = detector.detect(image_path)

    if not detections:
        print("  EasyOCR vide → fallback Florence-2...")
        try:
            florence = FlorenceDetector()
            detections = florence.detect(image_path)
        except Exception as e:
            print(f"  Florence-2 indisponible : {e}")

    matcher = LabelMatcher(labels_path)
    ocr_matches = matcher.match(detections)

    # Score OCR par salle : meilleure confiance parmi les matches fiables
    ocr_scores = {}
    strong = [m for m in ocr_matches if m["confidence"] >= OCR_MIN_CONFIDENCE]
    for m in strong:
        room = m["room"]
        if room not in ocr_scores or m["confidence"] > ocr_scores[room]:
            ocr_scores[room] = m["confidence"]

    return clip_result, ocr_scores, ocr_matches, detections


def fuse(clip_scores, ocr_scores, clip_w):
    """Fusionne les scores avec un poids clip_w pour CLIP et (1-clip_w) pour OCR."""
    ocr_w = 1.0 - clip_w
    rooms = set(clip_scores.keys()) | set(ocr_scores.keys())
    fused = {}
    for room in rooms:
        cs = clip_scores.get(room, 0.0)
        os_val = ocr_scores.get(room, 0.0)
        if os_val > 0:
            fused[room] = clip_w * cs + ocr_w * os_val
        else:
            fused[room] = clip_w * cs
    return fused


def analyze(index_path, labels_path, image_path, true_room=None, steps=101):
    clip_result, ocr_scores, ocr_matches, detections = get_raw_scores(
        index_path, labels_path, image_path
    )
    clip_scores = clip_result["room_scores"]
    rooms = sorted(set(clip_scores.keys()) | set(ocr_scores.keys()))

    weights = np.linspace(0, 1, steps)  # poids CLIP de 0% à 100%

    # Score final par poids pour chaque salle
    room_curves = {r: [] for r in rooms}
    best_rooms = []
    best_scores = []
    margins = []

    for w in weights:
        fused = fuse(clip_scores, ocr_scores, w)
        sorted_scores = sorted(fused.values(), reverse=True)
        best = max(fused, key=fused.get)
        best_rooms.append(best)
        best_scores.append(fused[best])
        margins.append(sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0)
        for r in rooms:
            room_curves[r].append(fused.get(r, 0.0))

    best_scores = np.array(best_scores)
    margins = np.array(margins)

    # Poids optimal = maximise score × marge
    combined = best_scores * margins
    opt_idx = int(np.argmax(combined))
    opt_w_clip = float(weights[opt_idx])
    opt_w_ocr = 1.0 - opt_w_clip

    # ── Affichage terminal ──────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ANALYSE DES POIDS — {Path(image_path).name}")
    print(sep)
    print(f"\n  Textes détectés ({len(detections)}) :")
    for d in detections:
        print(f"    [{d.get('source','?')}] '{d['text']}'  conf={d['confidence']:.2f}")

    if ocr_matches:
        print(f"\n  Matches OCR ({len(ocr_matches)}) :")
        for m in ocr_matches:
            flag = " ✓" if m["confidence"] >= OCR_MIN_CONFIDENCE else " (faible)"
            print(f"    '{m['text']}' → {m['room']}  [{m['match_type']}]  conf={m['confidence']:.2f}{flag}")
    else:
        print("\n  Aucun label reconnu par OCR.")

    ocr_used = bool(ocr_scores)
    print(f"\n  OCR actif dans la fusion : {'OUI' if ocr_used else 'NON (confiance trop faible)'}")

    print(f"\n  Poids optimal trouvé :")
    print(f"    CLIP = {opt_w_clip*100:.0f}%   OCR = {opt_w_ocr*100:.0f}%")
    print(f"    Score max    = {best_scores[opt_idx]:.4f}")
    print(f"    Marge        = {margins[opt_idx]:.4f}")
    print(f"    Salle prédite: {best_rooms[opt_idx]}")
    if true_room:
        correct = best_rooms[opt_idx] == true_room
        print(f"    Salle réelle : {true_room}  → {'CORRECT ✓' if correct else 'INCORRECT ✗'}")
    print(sep)

    # ── Graphique ───────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Optimisation des poids — {Path(image_path).name}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = plt.cm.tab10(np.linspace(0, 1, len(rooms)))
    x_pct = weights * 100  # axe en %

    # ── Graphe 1 : score par salle en fonction du poids CLIP ────
    ax1 = fig.add_subplot(gs[0, :])
    for i, room in enumerate(rooms):
        ax1.plot(x_pct, room_curves[room], label=room, color=colors[i], linewidth=2)

    ax1.axvline(opt_w_clip * 100, color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal CLIP={opt_w_clip*100:.0f}%")
    ax1.scatter([opt_w_clip * 100], [best_scores[opt_idx]],
                color="red", zorder=5, s=80)
    ax1.set_xlabel("Poids CLIP (%)", fontsize=10)
    ax1.set_ylabel("Score fusionné", fontsize=10)
    ax1.set_title("Score de chaque salle selon le poids CLIP", fontsize=11)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)

    # Annotation du point optimal
    ax1.annotate(
        f"  Optimal\n  CLIP={opt_w_clip*100:.0f}% / OCR={opt_w_ocr*100:.0f}%\n  score={best_scores[opt_idx]:.3f}",
        xy=(opt_w_clip * 100, best_scores[opt_idx]),
        xytext=(opt_w_clip * 100 + 5, best_scores[opt_idx] - 0.05),
        fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
    )

    # ── Graphe 2 : score maximal ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_pct, best_scores, color="royalblue", linewidth=2)
    ax2.axvline(opt_w_clip * 100, color="red", linestyle="--", linewidth=1.5)
    ax2.scatter([opt_w_clip * 100], [best_scores[opt_idx]], color="red", zorder=5, s=80)
    ax2.set_xlabel("Poids CLIP (%)", fontsize=10)
    ax2.set_ylabel("Score du meilleur match", fontsize=10)
    ax2.set_title("Score maximal vs poids CLIP", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    # ── Graphe 3 : marge (score1 - score2) ───────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x_pct, margins, color="darkorange", linewidth=2)
    ax3.axvline(opt_w_clip * 100, color="red", linestyle="--", linewidth=1.5)
    ax3.scatter([opt_w_clip * 100], [margins[opt_idx]], color="red", zorder=5, s=80)
    ax3.set_xlabel("Poids CLIP (%)", fontsize=10)
    ax3.set_ylabel("Marge (score1 − score2)", fontsize=10)
    ax3.set_title("Marge de décision vs poids CLIP", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)

    # Scores bruts dans un encadré
    info = (
        f"Scores CLIP bruts :\n"
        + "\n".join(f"  {r}: {clip_scores.get(r, 0):.4f}" for r in rooms)
        + (f"\n\nScores OCR bruts :\n"
           + "\n".join(f"  {r}: {ocr_scores[r]:.4f}" for r in ocr_scores)
           if ocr_scores else "\n\nOCR : aucun match fiable")
    )
    fig.text(0.01, 0.01, info, fontsize=7.5, verticalalignment="bottom",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    out_path = Path(image_path).stem + "_weight_analysis.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\n  Graphique sauvegardé : {out_path}")
    plt.show()


def analyze_test_dir(index_path, labels_path, test_dir, steps=101):
    """
    Mode jeu de test : calcule la précision pour chaque poids sur toutes les images étiquetées.
    Affiche le poids optimal global et la courbe de précision.

    Structure attendue :
        test_dir/
            salle1/photo1.jpg ...
            salle3/photo2.jpg ...
    """
    from pathlib import Path
    IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    test_dir = Path(test_dir)
    samples = []  # [(image_path, true_room), ...]
    for room_dir in sorted(test_dir.iterdir()):
        if not room_dir.is_dir():
            continue
        for img in sorted(room_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXT:
                samples.append((str(img), room_dir.name))

    if not samples:
        print(f"Aucune image trouvée dans {test_dir}")
        return

    print(f"\n  {len(samples)} images de test dans {len(set(s[1] for s in samples))} salles")

    weights = np.linspace(0, 1, steps)

    # Pour chaque image, récupère les scores bruts une seule fois
    all_clip = []
    all_ocr  = []
    all_true = []

    for i, (img_path, true_room) in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {Path(img_path).name} ({true_room})")
        try:
            clip_result, ocr_scores, _, _ = get_raw_scores(index_path, labels_path, img_path)
            all_clip.append(clip_result["room_scores"])
            all_ocr.append(ocr_scores)
            all_true.append(true_room)
        except Exception as e:
            print(f"    Erreur : {e}")

    if not all_clip:
        print("Aucun résultat.")
        return

    # Précision par poids
    accuracy_clip_only = []
    accuracy_hybrid    = []
    accuracy_ocr_only  = []

    for w in weights:
        correct_hybrid    = 0
        correct_clip_only = 0
        correct_ocr_only  = 0

        for clip_s, ocr_s, true in zip(all_clip, all_ocr, all_true):
            # CLIP seul
            best_clip = max(clip_s, key=clip_s.get)
            if best_clip == true:
                correct_clip_only += 1

            # Hybride
            fused = fuse(clip_s, ocr_s, w)
            if fused and max(fused, key=fused.get) == true:
                correct_hybrid += 1

            # OCR seul (w=0)
            fused_ocr = fuse(clip_s, ocr_s, 0.0)
            if fused_ocr and max(fused_ocr, key=fused_ocr.get) == true:
                correct_ocr_only += 1

        n = len(all_true)
        accuracy_hybrid.append(correct_hybrid / n * 100)
        accuracy_clip_only.append(correct_clip_only / n * 100)
        accuracy_ocr_only.append(correct_ocr_only / n * 100)

    accuracy_hybrid    = np.array(accuracy_hybrid)
    accuracy_clip_only = np.array(accuracy_clip_only)

    opt_idx   = int(np.argmax(accuracy_hybrid))
    opt_w     = float(weights[opt_idx])
    opt_acc   = accuracy_hybrid[opt_idx]
    acc_clip  = accuracy_clip_only[0]
    acc_ocr   = accuracy_hybrid[0]   # w=0 → 100% OCR

    # ── Terminal ─────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RÉSULTATS SUR {len(all_true)} IMAGES")
    print(sep)
    print(f"  CLIP seul (100%)      : {acc_clip:.1f}% de précision")
    print(f"  OCR seul  (100%)      : {acc_ocr:.1f}% de précision")
    print(f"  Poids optimal hybride : CLIP={opt_w*100:.0f}% / OCR={(1-opt_w)*100:.0f}%")
    print(f"  Précision optimale    : {opt_acc:.1f}%")
    gain_clip = opt_acc - acc_clip
    gain_ocr  = opt_acc - acc_ocr
    print(f"  Gain vs CLIP seul     : {gain_clip:+.1f}%")
    print(f"  Gain vs OCR seul      : {gain_ocr:+.1f}%")
    if gain_clip > 0:
        print(f"\n  → L'OCR améliore la précision de {gain_clip:.1f}% avec CLIP={opt_w*100:.0f}%")
    else:
        print(f"\n  → CLIP seul est suffisant sur ce jeu de test ({gain_clip:.1f}% de gain OCR)")
    print(sep)

    # ── Graphique ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Optimisation des poids — {len(all_true)} images de test", fontsize=13, fontweight="bold")
    x_pct = weights * 100

    # Courbe de précision
    ax = axes[0]
    ax.plot(x_pct, accuracy_hybrid, color="royalblue", linewidth=2.5, label="Hybride (CLIP+OCR)")
    ax.axhline(acc_clip, color="green",  linestyle="--", linewidth=1.5, label=f"CLIP seul : {acc_clip:.1f}%")
    ax.axhline(acc_ocr,  color="orange", linestyle="--", linewidth=1.5, label=f"OCR seul  : {acc_ocr:.1f}%")
    ax.axvline(opt_w * 100, color="red", linestyle="--", linewidth=1.5, label=f"Optimal : CLIP={opt_w*100:.0f}%")
    ax.scatter([opt_w * 100], [opt_acc], color="red", zorder=5, s=100)
    ax.annotate(
        f"  CLIP={opt_w*100:.0f}% / OCR={(1-opt_w)*100:.0f}%\n  précision={opt_acc:.1f}%",
        xy=(opt_w * 100, opt_acc),
        xytext=(min(opt_w * 100 + 8, 70), opt_acc - 8),
        fontsize=9, color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )
    ax.set_xlabel("Poids CLIP (%)", fontsize=10)
    ax.set_ylabel("Précision (%)", fontsize=10)
    ax.set_title("Précision selon le poids CLIP", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)

    # Gain OCR vs CLIP seul
    ax2 = axes[1]
    gain = accuracy_hybrid - acc_clip
    colors_gain = ["green" if g >= 0 else "red" for g in gain]
    ax2.bar(x_pct, gain, width=100/steps, color=colors_gain, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.axvline(opt_w * 100, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Poids CLIP (%)", fontsize=10)
    ax2.set_ylabel("Gain de précision vs CLIP seul (%)", fontsize=10)
    ax2.set_title("Apport de l'OCR selon le poids", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    plt.tight_layout()
    out_path = "weight_analysis_testset.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\n  Graphique sauvegardé : {out_path}")
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Optimisation des poids CLIP/OCR")
    p.add_argument("--index",    required=True, help="index.pkl")
    p.add_argument("--labels",   required=True, help="labels.json")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",    help="image unique à analyser")
    mode.add_argument("--test-dir", help="dossier de test étiqueté (salle/photo.jpg)")

    p.add_argument("--true-room", default=None, help="salle réelle (mode --image uniquement)")
    p.add_argument("--steps", type=int, default=101, help="nombre de points sur la courbe")
    args = p.parse_args()

    if args.image:
        analyze(args.index, args.labels, args.image,
                true_room=args.true_room, steps=args.steps)
    else:
        analyze_test_dir(args.index, args.labels, args.test_dir, steps=args.steps)


if __name__ == "__main__":
    main()
