"""
Diagnostic des images de référence pour la localisation.
Lancer depuis la racine du projet :
    python diagnose_images.py --ref-dir img_mapanything
"""
import argparse, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ref-dir", required=True)
args = parser.parse_args()

ref_dir = Path(args.ref_dir)
images  = sorted(ref_dir.glob("*"))
images  = [p for p in images if p.suffix.lower() in (".jpg",".jpeg",".png",".heic",".tiff",".bmp",".webp")]

print(f"\n📂  {ref_dir.resolve()}")
print(f"    {len(images)} fichier(s) image trouvé(s)\n")

for p in images[:5]:   # tester les 5 premiers
    print(f"── {p.name}")

    # 1. Premiers octets (magic bytes = vrai format)
    with open(p, "rb") as f:
        header = f.read(12)
    hex_header = " ".join(f"{b:02x}" for b in header)
    print(f"   Magic bytes : {hex_header}")
    if header[:4] == b'\x89PNG':
        print("   → PNG standard ✓")
    elif header[:4] == b'\x89PNG' and header[12:16] == b'CgBI':
        print("   → PNG Apple crushed (CgBI) ← incompatible OpenCV/PIL standard")
    elif header[:4] in (b'ftypheic', b'\x00\x00\x00') :
        print("   → Probablement HEIC")
    elif header[:2] == b'\xff\xd8':
        print("   → JPEG ✓")
    elif header[4:8] == b'ftyp':
        print("   → HEIC/MP4 (container ISO)")
    else:
        print("   → Format inconnu")

    # 2. Test cv2
    try:
        import cv2, numpy as np
        raw = p.read_bytes()
        arr = np.asarray(bytearray(raw), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"   cv2.imdecode  ✓  ({img.shape[1]}×{img.shape[0]})")
        else:
            print("   cv2.imdecode  ✗  (retourne None)")
    except Exception as e:
        print(f"   cv2.imdecode  ✗  Exception : {e}")

    # 3. Test PIL
    try:
        from PIL import Image
        import io
        raw = p.read_bytes()
        pil = Image.open(io.BytesIO(raw))
        print(f"   PIL.open      ✓  mode={pil.mode}  taille={pil.size}")
        gray = pil.convert("L")
        print(f"   PIL.convert   ✓")
    except Exception as e:
        print(f"   PIL.open      ✗  Exception : {e}")

    # 4. Test imageio (si disponible)
    try:
        import imageio
        import numpy as np
        img = imageio.imread(str(p), as_gray=True)
        print(f"   imageio       ✓  shape={img.shape}")
    except ImportError:
        print("   imageio       – non installé (pip install imageio)")
    except Exception as e:
        print(f"   imageio       ✗  {e}")

    print()

print("Installe les dépendances manquantes si besoin :")
print("  pip install pillow-heif   # pour HEIC")
print("  pip install imageio       # alternative universelle")
