"""
Localisation visuelle pour MapAnything Navigation
==================================================
Estime la position X,Y,Z du robot à partir d'une image caméra
en la comparant à la base de données d'images de la reconstruction.

Algorithme SIFT + FLANN :
  1. Au démarrage : extraction SIFT sur toutes les images de référence
  2. Pour chaque frame requête : matching FLANN + test de ratio de Lowe
  3. Position = moyenne pondérée par le nombre de bons matchs des top-5 références

Sources de poses supportées (auto-détectées dans --ref-dir) :
  ① images.txt  – export COLMAP depuis colmap_export.py  (le plus précis)
  ② poses.json  – format simple : {"image.jpg": [x, y, z], ...}

Dépendance : pip install opencv-python
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("⚠️  opencv-python non installé : pip install opencv-python")

# Support HEIC/HEIF (iPhone) — sans pillow-heif ces fichiers sont illisibles
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # pillow-heif optionnel ; pip install pillow-heif pour les images iPhone


# ──────────────────────────────────────────────────────────
#  Utilitaires COLMAP
# ──────────────────────────────────────────────────────────

def _quat_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (w,x,y,z) → matrice de rotation 3×3."""
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def _parse_colmap_images(images_txt: Path, images_dir: Path) -> List[Dict]:
    """
    Parse COLMAP images.txt et retourne les références avec position caméra.
    Format : IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME  (1 ligne sur 2)
    Position caméra dans le monde : C = -R^T @ t
    """
    refs = []
    if not images_txt.exists():
        return refs

    with open(images_txt) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    i = 0
    while i < len(lines) - 1:
        parts = lines[i].split()
        if len(parts) < 10:
            i += 1
            continue
        try:
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz     = map(float, parts[5:8])
            name = parts[9]
        except (ValueError, IndexError):
            i += 2
            continue

        R   = _quat_to_rotation(qw, qx, qy, qz)
        pos = -R.T @ np.array([tx, ty, tz])   # position caméra dans le repère monde

        img_path = images_dir / name
        if not img_path.exists():
            # Chercher dans les sous-répertoires
            candidates = list(images_dir.rglob(name))
            img_path = candidates[0] if candidates else img_path

        if img_path.exists():
            refs.append({"filename": name, "path": img_path, "pos": pos})
        i += 2
    return refs


def _parse_poses_json(poses_json: Path) -> List[Dict]:
    """
    Parse poses.json : {"image.jpg": [x, y, z], ...}
    Les images doivent être dans le même répertoire que poses.json.
    """
    if not poses_json.exists():
        return []
    ref_dir = poses_json.parent
    with open(poses_json) as f:
        data = json.load(f)
    return [
        {"filename": fname,
         "path": ref_dir / fname,
         "pos": np.array(pos, dtype=np.float64)}
        for fname, pos in data.items()
        if (ref_dir / fname).exists()
    ]


# ──────────────────────────────────────────────────────────
#  Classe principale
# ──────────────────────────────────────────────────────────

class VisualLocalizer:
    """
    Localisation visuelle par matching de descripteurs SIFT.

    Paramètres
    ----------
    ref_dir : chemin vers le répertoire contenant les images de référence
              et le fichier de poses (images.txt ou poses.json).
    """

    def __init__(self, ref_dir: Optional[str] = None):
        self.ref_dir   = Path(ref_dir) if ref_dir else None
        self.db: List[Dict] = []     # [{filename, pos, descriptors}, ...]
        self.ready     = False
        self.error_msg = ""

        if CV2_OK and self.ref_dir and self.ref_dir.exists():
            self._build_database()
        elif not CV2_OK:
            self.error_msg = "opencv-python non installé : pip install opencv-python"
        elif not self.ref_dir:
            self.error_msg = (
                "Aucun répertoire de référence fourni. "
                "Lancer le serveur avec --ref-dir <dossier>."
            )

    # ── Utilitaire de chargement d'image ────────────────────

    @staticmethod
    def _load_gray(path: Path) -> Optional[np.ndarray]:
        """
        Charge une image en niveaux de gris depuis le disque.

        Sur macOS, cv2.imread échoue silencieusement sur les PNG iPhone
        (profil couleur Display P3, bit-depth non standard…).
        On contourne en lisant les bytes bruts puis en appelant cv2.imdecode,
        ce qui bypasse la couche filesystem de OpenCV.
        Fallback PIL pour les formats non gérés par OpenCV (HEIC via pillow-heif…).
        """
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            return None

        # cv2.imdecode sur bytes (bytearray = buffer accessible en écriture)
        arr = np.asarray(bytearray(raw), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img

        # Fallback PIL (HEIC avec pillow-heif, TIFF, etc.)
        try:
            import io as _io
            from PIL import Image as _PIL
            pil = _PIL.open(_io.BytesIO(raw)).convert("L")
            return np.array(pil, dtype=np.uint8)
        except Exception:
            return None

    # ── Construction de la base de données ──────────────────

    def _build_database(self) -> None:
        """
        1. Charge les poses depuis images.txt (COLMAP) ou poses.json
        2. Extrait les descripteurs SIFT de chaque image référence
        """
        # Détection automatique du format
        refs_raw: List[Dict] = []

        colmap_txt = self.ref_dir / "images.txt"
        img_subdir = self.ref_dir / "images"
        img_dir    = img_subdir if img_subdir.exists() else self.ref_dir

        if colmap_txt.exists():
            refs_raw = _parse_colmap_images(colmap_txt, img_dir)
            print(f"  Localizer : {len(refs_raw)} images depuis COLMAP images.txt")

        if not refs_raw:
            poses_json = self.ref_dir / "poses.json"
            refs_raw   = _parse_poses_json(poses_json)
            print(f"  Localizer : {len(refs_raw)} images depuis poses.json")

        if not refs_raw:
            self.error_msg = (
                f"Aucune pose trouvée dans {self.ref_dir}.\n"
                "Fournir images.txt (COLMAP) ou poses.json."
            )
            print(f"  ⚠️  {self.error_msg}")
            return

        sift     = cv2.SIFT_create(nfeatures=2000)
        loaded   = 0
        n_no_img = 0
        n_no_kp  = 0

        for ref in refs_raw:
            img = self._load_gray(ref["path"])
            if img is None:
                n_no_img += 1
                print(f"    ✗ Impossible de lire : {ref['filename']}")
                continue

            # Redimensionner si trop grande (accélère le matching)
            h, w = img.shape
            if max(h, w) > 1024:
                s   = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))

            kp, des = sift.detectAndCompute(img, None)
            if des is None or len(des) < 5:   # seuil abaissé à 5
                n_no_kp += 1
                print(f"    ✗ Trop peu de keypoints ({len(des) if des is not None else 0}) : {ref['filename']}")
                continue

            self.db.append({
                "filename":    ref["filename"],
                "pos":         ref["pos"],
                "descriptors": des.astype(np.float32),
            })
            loaded += 1

        self.ready = loaded > 0
        if self.ready:
            print(f"  ✓ Localizer prêt : {loaded}/{len(refs_raw)} images chargées"
                  + (f"  ({n_no_img} illisibles, {n_no_kp} sans keypoints)" if n_no_img+n_no_kp else ""))
        else:
            detail = []
            if n_no_img:  detail.append(f"{n_no_img} image(s) illisible(s) par OpenCV")
            if n_no_kp:   detail.append(f"{n_no_kp} image(s) sans keypoints SIFT")
            self.error_msg = (
                f"Aucune image valide sur {len(refs_raw)} trouvées"
                + (f" : {', '.join(detail)}" if detail else "")
                + ". Vérifier le terminal pour le détail."
            )
            print(f"  ⚠️  {self.error_msg}")

    # ── Matching ─────────────────────────────────────────────

    def _count_good_matches(self, des_q: np.ndarray, des_ref: np.ndarray) -> int:
        """
        Nombre de bonnes correspondances SIFT via FLANN + test de ratio de Lowe (0.75).
        """
        FLANN_INDEX_KDTREE = 1
        flann = cv2.FlannBasedMatcher(
            {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
            {"checks": 50},
        )
        try:
            matches = flann.knnMatch(des_q, des_ref, k=2)
        except cv2.error:
            return 0
        return sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance)

    # ── Localisation d'une frame ──────────────────────────────

    def _localize_frame(self, img_gray: np.ndarray) -> Dict:
        """
        Localise une image (tableau numpy niveaux de gris).
        Retourne {x, y, z, confidence, best_match, match_count}.
        """
        if not self.ready:
            return {"error": self.error_msg or "Base de données non prête."}

        # Resize
        h, w = img_gray.shape
        if max(h, w) > 1024:
            s       = 1024 / max(h, w)
            img_gray = cv2.resize(img_gray, (int(w * s), int(h * s)))

        sift     = cv2.SIFT_create(nfeatures=2000)
        _, des_q = sift.detectAndCompute(img_gray, None)
        if des_q is None or len(des_q) < 10:
            return {"error": "Pas assez de points caractéristiques dans l'image."}
        des_q = des_q.astype(np.float32)

        # Score pour chaque référence
        scores = sorted(
            [(self._count_good_matches(des_q, r["descriptors"]), r) for r in self.db],
            key=lambda x: x[0], reverse=True,
        )

        top_k     = scores[:min(5, len(scores))]
        total_w   = sum(s for s, _ in top_k)

        if total_w == 0:
            pos = top_k[0][1]["pos"]
            confidence = 0.0
        else:
            pos_arr = np.zeros(3)
            for cnt, ref in top_k:
                pos_arr += cnt * ref["pos"]
            pos = pos_arr / total_w
            # 80 bons matchs → confiance maximale
            confidence = float(min(1.0, top_k[0][0] / 80.0))

        return {
            "x":           round(float(pos[0]), 4),
            "y":           round(float(pos[1]), 4),
            "z":           round(float(pos[2]), 4),
            "confidence":  round(confidence, 3),
            "best_match":  top_k[0][1]["filename"],
            "match_count": int(top_k[0][0]),
        }

    # ── API publiques ────────────────────────────────────────

    def localize_image_bytes(self, image_bytes: bytes) -> Dict:
        """Localise depuis les bytes bruts d'une image (JPEG, PNG…)."""
        if not CV2_OK:
            return {"error": "opencv-python non installé : pip install opencv-python"}

        # np.frombuffer crée un tableau read-only que cv2.imdecode refuse parfois.
        # bytearray() force une copie accessible en écriture.
        arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback PIL pour les formats moins courants (HEIC, TIFF, RGBA PNG…)
            try:
                import io as _io
                from PIL import Image as _PILImage
                pil = _PILImage.open(_io.BytesIO(image_bytes)).convert("L")
                img = np.array(pil, dtype=np.uint8)
            except Exception as e:
                return {"error": f"Image non décodable : {e}"}

        if img is None:
            return {"error": "Image non décodable — essayez JPEG ou PNG standard."}

        return self._localize_frame(img)

    def localize_video_bytes(
        self,
        video_bytes: bytes,
        sample_fps: float = 1.0,
        max_frames: int   = 60,
    ) -> List[Dict]:
        """
        Localise depuis les bytes d'une vidéo.

        Paramètres
        ----------
        sample_fps : frames par seconde à analyser (défaut : 1)
        max_frames : nombre max de frames analysées (défaut : 60)

        Retourne
        --------
        Liste de {frame, time_s, x, y, z, confidence, best_match, match_count}
        """
        if not CV2_OK:
            return [{"error": "opencv-python non installé"}]

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(video_bytes)
        tmp.close()

        results: List[Dict] = []
        try:
            cap      = cv2.VideoCapture(tmp.name)
            src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval = max(1, int(src_fps / sample_fps))
            frame_idx = 0

            while len(results) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    loc  = self._localize_frame(gray)
                    loc["frame"]  = frame_idx
                    loc["time_s"] = round(frame_idx / src_fps, 2)
                    results.append(loc)
                frame_idx += 1
            cap.release()
        finally:
            os.unlink(tmp.name)

        return results

    def get_status(self) -> Dict:
        return {
            "ready":            self.ready,
            "opencv_available": CV2_OK,
            "reference_count":  len(self.db),
            "ref_dir":          str(self.ref_dir) if self.ref_dir else None,
            "error":            self.error_msg if not self.ready else None,
        }
