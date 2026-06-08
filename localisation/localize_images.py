"""
Room Localization — Methode 1 : Image Retrieval
================================================

Compare une photo directement aux images ORIGINALES de chaque salle
en utilisant des embeddings CNN (ResNet / EfficientNet / CLIP).

Pourquoi c'est mieux que comparer au PLY :
  - Photo ↔ Photo = pas de gap de luminosite
  - Les CNN capturent la semantique (meubles, murs, objets)
  - Embeddings pre-entraines = aucun entrainement necessaire
  - Rejet fiable des images hors-batiment

Entrees :
  - Dossier avec sous-dossiers par salle contenant les JPG originaux
  - Une photo query prise par l'utilisateur

Usage :
    python localize_images.py index --rooms-dir ./mon_batiment --output index.pkl
    python localize_images.py query --index index.pkl --image photo.jpg
    python localize_images.py serve --index index.pkl --port 8000

Structure attendue :
    mon_batiment/
        salle_101/
            frame_001.jpg
            frame_002.jpg
            ...
        salle_102/
            frame_001.jpg
            ...
        couloir_A/
            frame_001.jpg
            ...
"""

import os
import sys
import json
import pickle
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ═══════════════════════════════════════════════════════
#  IMAGE EMBEDDER — CNN feature extraction
# ═══════════════════════════════════════════════════════

class ImageEmbedder:
    """
    Extrait un embedding (vecteur dense) d'une image via un CNN pre-entraine.
    
    Backends supportes (par ordre de preference) :
    1. CLIP (open_clip) — le meilleur, embeddings semantiques
    2. EfficientNet — bon compromis vitesse/qualite
    3. ResNet50 — classique, robuste
    4. ResNet18 — plus leger
    
    Le premier backend disponible est utilise automatiquement.
    """

    def __init__(self, backend: str = "auto", device: str = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.model = None
        self.transform = None
        self.feat_dim = 0

        if backend == "auto":
            # Essayer dans l'ordre de preference
            for b in ["clip", "efficientnet_b0", "resnet50", "resnet18"]:
                try:
                    self._load_backend(b)
                    self.backend = b
                    print(f"  Backend: {b} ({self.feat_dim}D) sur {self.device}")
                    return
                except Exception:
                    continue
            raise RuntimeError("Aucun backend disponible. Installe torch + torchvision.")
        else:
            self._load_backend(backend)
            print(f"  Backend: {backend} ({self.feat_dim}D) sur {self.device}")

    def _load_backend(self, name: str):
        import torch
        import torch.nn as nn

        if name == "clip":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.model = model.to(self.device).eval()
            self.transform = preprocess
            self.feat_dim = 512
            self._extract_fn = self._extract_clip

        elif name == "efficientnet_b0":
            from torchvision import models, transforms
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.model = nn.Sequential(base.features, base.avgpool, nn.Flatten()).to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.feat_dim = 1280
            self._extract_fn = self._extract_torchvision

        elif name == "resnet50":
            from torchvision import models, transforms
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model = nn.Sequential(*list(base.children())[:-1], nn.Flatten()).to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.feat_dim = 2048
            self._extract_fn = self._extract_torchvision

        elif name == "resnet18":
            from torchvision import models, transforms
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.model = nn.Sequential(*list(base.children())[:-1], nn.Flatten()).to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.feat_dim = 512
            self._extract_fn = self._extract_torchvision

        else:
            raise ValueError(f"Backend inconnu: {name}")

    @staticmethod
    def _load_image(path):
        return Image.open(str(path)).convert("RGB")

    def _extract_clip(self, img: Image.Image) -> np.ndarray:
        import torch
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten()

    def _extract_torchvision(self, img: Image.Image) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).flatten()
            feat = F.normalize(feat, dim=0)
        return feat.cpu().numpy()

    def extract(self, image) -> np.ndarray:
        """Extraire l'embedding d'une image (path, PIL, ou numpy array)."""
        if isinstance(image, (str, Path)):
            img = self._load_image(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = image.convert("RGB")
        return self._extract_fn(img)

    def extract_batch(self, paths: list, batch_size: int = 32) -> np.ndarray:
        """Extraire les embeddings pour une liste d'images."""
        import torch
        all_feats = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                try:
                    img = self._load_image(p)
                    tensors.append(self.transform(img))
                except Exception as e:
                    print(f"    ⚠ Skip {p}: {e}")
                    continue

            if not tensors:
                continue

            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                if self.backend == "clip":
                    feats = self.model.encode_image(batch)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                else:
                    feats = self.model(batch)
                    feats = torch.nn.functional.normalize(feats, dim=1)
            all_feats.append(feats.cpu().numpy())

        return np.vstack(all_feats) if all_feats else np.zeros((0, self.feat_dim))


# ═══════════════════════════════════════════════════════
#  ROOM INDEX
# ═══════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


class RoomIndex:
    """
    Index de localisation base sur les images originales.
    Chaque salle = un dossier d'images.
    """

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None  # (N_images, D)
        self.metadata: list = []   # [{room, image_name, image_path}, ...]
        self.rooms: list = []
        self.room_image_counts: dict = {}
        self.backend: str = ""
        self.feat_dim: int = 0
        self.rejection_threshold: float = 0.0

    def build(self, rooms_dir: str, backend: str = "auto", max_images_per_room: int = 500):
        """
        Indexe les images de chaque salle.

        Structure :
            rooms_dir/
                salle_A/
                    img1.jpg, img2.jpg, ...
                salle_B/
                    img1.jpg, ...
        """
        rooms_dir = Path(rooms_dir)
        room_dirs = sorted([
            d for d in rooms_dir.iterdir()
            if d.is_dir() and not d.name.startswith('_') and not d.name.startswith('.')
        ])

        if not room_dirs:
            raise FileNotFoundError(
                f"Aucun sous-dossier trouvé dans {rooms_dir}.\n"
                f"Structure attendue : {rooms_dir}/salle_A/*.jpg, {rooms_dir}/salle_B/*.jpg, ..."
            )

        # Collecter toutes les images
        all_paths = []
        all_meta = []

        print(f"\n{'='*60}")
        print(f"  INDEXATION — Image Retrieval (Methode 1)")
        print(f"{'='*60}\n")

        for room_dir in room_dirs:
            room_name = room_dir.name
            images = sorted([
                f for f in room_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ])

            if not images:
                print(f"  ⚠ {room_name}: aucune image, skip")
                continue

            # Limiter le nombre d'images par salle
            if len(images) > max_images_per_room:
                step = len(images) // max_images_per_room
                images = images[::step][:max_images_per_room]

            print(f"  📁 {room_name}: {len(images)} images")

            for img_path in images:
                all_paths.append(str(img_path))
                all_meta.append({
                    "room": room_name,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                })

            self.room_image_counts[room_name] = len(images)

        if not all_paths:
            raise ValueError("Aucune image trouvée dans aucune salle!")

        total = len(all_paths)
        print(f"\n  📊 Total: {total} images dans {len(self.room_image_counts)} salles")

        # Extraire les embeddings
        print(f"  ⏳ Extraction des embeddings...")
        embedder = ImageEmbedder(backend=backend)
        self.backend = embedder.backend
        self.feat_dim = embedder.feat_dim

        t0 = time.time()
        self.embeddings = embedder.extract_batch(all_paths)
        elapsed = time.time() - t0

        print(f"  ✅ {len(self.embeddings)} embeddings en {elapsed:.1f}s ({len(self.embeddings)/max(elapsed,0.01):.0f} img/s)")

        self.metadata = all_meta[:len(self.embeddings)]
        self.rooms = sorted(self.room_image_counts.keys())

        # Calculer le seuil de rejet
        self._compute_rejection_threshold()

        print(f"\n  📐 Embeddings: {self.embeddings.shape}")
        print(f"  🚫 Seuil de rejet: {self.rejection_threshold:.4f}")

    def _compute_rejection_threshold(self):
        """Seuil base sur les distances inter-salles."""
        if len(self.rooms) > 1:
            # Calculer la similarite moyenne de chaque salle
            room_centroids = {}
            for room in self.rooms:
                mask = np.array([m["room"] == room for m in self.metadata])
                room_centroids[room] = self.embeddings[mask].mean(axis=0)
                room_centroids[room] /= np.linalg.norm(room_centroids[room]) + 1e-8

            # Similarite inter-salles
            inter_sims = []
            rooms_list = list(room_centroids.keys())
            for i in range(len(rooms_list)):
                for j in range(i + 1, len(rooms_list)):
                    sim = np.dot(room_centroids[rooms_list[i]], room_centroids[rooms_list[j]])
                    inter_sims.append(sim)

            mean_inter = np.mean(inter_sims)
            self.rejection_threshold = max(mean_inter - 0.05, 0.3)
        else:
            # Une seule salle — seuil basé sur la cohérence interne
            sims = self.embeddings @ self.embeddings.T
            np.fill_diagonal(sims, 0)
            mean_self = sims.mean()
            self.rejection_threshold = mean_self * 0.5

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "metadata": self.metadata,
                "rooms": self.rooms,
                "room_image_counts": self.room_image_counts,
                "backend": self.backend,
                "feat_dim": self.feat_dim,
                "rejection_threshold": self.rejection_threshold,
                "version": "method1_v1",
            }, f)
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  💾 Sauvegardé: {path} ({mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "RoomIndex":
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx = cls()
        idx.embeddings = d["embeddings"]
        idx.metadata = d["metadata"]
        idx.rooms = d["rooms"]
        idx.room_image_counts = d.get("room_image_counts", {})
        idx.backend = d.get("backend", "resnet18")
        idx.feat_dim = d.get("feat_dim", idx.embeddings.shape[1])
        idx.rejection_threshold = d.get("rejection_threshold", 0.3)
        return idx

    def query(self, image_path: str, top_k: int = 10) -> dict:
        """
        Localise une photo parmi les salles indexees.
        
        Retourne la salle, la confiance, le top-K des images
        les plus proches, et si l'image est rejetee.
        """
        embedder = ImageEmbedder(backend=self.backend)
        query_emb = embedder.extract(image_path)

        # Similarite cosinus avec toutes les images indexees
        sims = self.embeddings @ query_emb

        # Top-K images les plus proches
        top_idx = np.argsort(sims)[::-1][:top_k]
        top_matches = []
        for i in top_idx:
            top_matches.append({
                "room": self.metadata[i]["room"],
                "image": self.metadata[i]["image_name"],
                "similarity": float(sims[i]),
            })

        # Score par salle : moyenne du top-5 des images de chaque salle
        room_scores = {}
        for room in self.rooms:
            mask = np.array([m["room"] == room for m in self.metadata])
            room_sims = sims[mask]
            n = min(5, len(room_sims))
            room_scores[room] = float(np.mean(np.sort(room_sims)[::-1][:n]))

        best_room = max(room_scores, key=room_scores.get)
        best_score = room_scores[best_room]

        sorted_scores = sorted(room_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0

        rejected = best_score < self.rejection_threshold

        return {
            "best_room": best_room if not rejected else "INCONNU",
            "confidence": best_score,
            "margin": float(margin),
            "room_scores": room_scores,
            "top_matches": top_matches,
            "rejected": rejected,
            "rejection_threshold": self.rejection_threshold,
            "best_match_image": top_matches[0]["image"] if top_matches else None,
        }


# ═══════════════════════════════════════════════════════
#  WEB UI — avec affichage de la salle correspondante
# ═══════════════════════════════════════════════════════

WEB_UI = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8"><title>Room Localization</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#07080a;color:#e6edf3;font-family:system-ui,sans-serif;min-height:100vh}
.app{max-width:720px;margin:0 auto;padding:32px 20px}
h1{font-size:20px;font-weight:500;margin-bottom:4px}
.sub{font-size:12px;color:#8b949e;margin-bottom:32px}
.drop{border:2px dashed #21262d;border-radius:12px;padding:48px;text-align:center;
  cursor:pointer;background:#0e1117;transition:all 0.2s;margin-bottom:24px}
.drop:hover{border-color:#58a6ff;background:rgba(88,166,255,0.05)}
.drop.has{padding:0;border-style:solid;overflow:hidden}
.drop img{width:100%;max-height:300px;object-fit:cover;display:block}
input[type=file]{display:none}
.ld{text-align:center;padding:32px;color:#8b949e}
.sp{width:24px;height:24px;border:2px solid #21262d;border-top-color:#58a6ff;
  border-radius:50%;margin:0 auto 8px;animation:s .6s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
.res{animation:fi .4s ease}
@keyframes fi{from{opacity:0;transform:translateY(10px)}to{opacity:1}}
.hero{display:flex;align-items:center;gap:20px;padding:24px;background:#0e1117;
  border:1px solid #21262d;border-radius:12px;margin-bottom:16px}
.ring{position:relative;flex-shrink:0}
.ring svg{transform:rotate(-90deg)}
.ring .v{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring .n{font-size:28px;font-weight:500;line-height:1}
.ring .u{font-size:9px;color:#484f58;letter-spacing:1px;margin-top:2px}
.nfo .lb{font-size:9px;color:#484f58;letter-spacing:1.5px;margin-bottom:4px}
.nfo .rm{font-size:22px;font-weight:500;margin-bottom:6px}
.nfo .st{font-size:11px;color:#8b949e}.nfo .st b{color:#e6edf3;font-weight:400}
.rej{background:rgba(248,81,73,0.08);border-color:rgba(248,81,73,0.3)}
.rej .n{color:#f85149!important}.rej .rm{color:#f85149}
.sec{padding:16px;background:#0e1117;border:1px solid #21262d;border-radius:10px;margin-bottom:12px}
.sec .tt{font-size:9px;color:#484f58;letter-spacing:1.5px;margin-bottom:10px}
.br{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:6px;margin-bottom:3px}
.br.w{background:rgba(63,185,80,0.08);border:1px solid rgba(63,185,80,0.15)}
.br .d{width:8px;height:8px;border-radius:2px;flex-shrink:0}
.br .nm{font-size:12px;flex:1}
.br .bg{font-size:8px;padding:1px 5px;border-radius:3px;background:rgba(63,185,80,0.15);color:#3fb950;margin-left:6px}
.br .tk{flex:2;height:4px;background:#161b22;border-radius:2px;overflow:hidden}
.br .fl{height:100%;border-radius:2px;transition:width .8s ease}
.br .sc{font-size:12px;min-width:50px;text-align:right;font-variant-numeric:tabular-nums}
.mr{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:5px;
  background:#161b22;margin-bottom:4px;font-size:11px}
.mr .rk{color:#484f58;min-width:20px}
.mr .dt{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.mr .inf{flex:1;color:#8b949e}.mr .inf strong{color:#e6edf3;font-weight:400}
.mr .sm{min-width:50px;text-align:right;font-variant-numeric:tabular-nums}
.rjb{margin-top:12px;padding:12px 16px;border-radius:8px;background:rgba(248,81,73,0.08);
  border:1px solid rgba(248,81,73,0.2);font-size:12px;color:#f85149;line-height:1.5}
</style></head><body>
<div class="app">
<h1>&#9678; Room localization</h1>
<p class="sub">Methode 1 — Image Retrieval (photo &#8596; photos originales)</p>
<div class="drop" id="dr" onclick="document.getElementById('f').click()">
<input type="file" id="f" accept="image/*" onchange="go(this.files[0])">
<div style="color:#8b949e;font-size:14px">Glisse une photo ici</div>
<div style="color:#484f58;font-size:11px;margin-top:4px">ou clique pour parcourir</div>
</div>
<div id="out"></div>
</div>
<script>
const dr=document.getElementById('dr');
dr.ondragover=e=>{e.preventDefault();dr.style.borderColor='#58a6ff'};
dr.ondragleave=()=>dr.style.borderColor='#21262d';
dr.ondrop=e=>{e.preventDefault();dr.style.borderColor='#21262d';go(e.dataTransfer.files[0])};
const C=['#58a6ff','#3fb950','#d29922','#f78166','#bc8cff','#f778ba','#79c0ff'];
let ci=0;const M={};function gc(r){if(!M[r])M[r]=C[ci++%C.length];return M[r]}

async function go(f){
  if(!f)return;
  dr.className='drop has';dr.innerHTML='<img src="'+URL.createObjectURL(f)+'">';
  const o=document.getElementById('out');
  o.innerHTML='<div class="ld"><div class="sp"></div>Analyse en cours...</div>';
  try{
    const fd=new FormData();fd.append('image',f);
    const r=await(await fetch('/localize',{method:'POST',body:fd})).json();
    render(r);
  }catch(e){o.innerHTML='<div class="rjb">Erreur: '+e.message+'</div>'}
}

function render(d){
  const b=d.best_room,c=d.confidence,rj=d.rejected,co=rj?'#f85149':gc(b);
  const p=Math.round(c*100),ci2=2*Math.PI*48,of=ci2-c*ci2;
  const sc=Object.entries(d.room_scores).sort((a,b)=>b[1]-a[1]),mx=sc[0]?sc[0][1]:1;
  const matches=d.top_matches||[];
  let h='<div class="res">';

  // Hero
  h+='<div class="hero'+(rj?' rej':'')+'"><div class="ring">';
  h+='<svg width="110" height="110"><circle cx="55" cy="55" r="48" fill="none" stroke="#21262d" stroke-width="6"/>';
  h+='<circle cx="55" cy="55" r="48" fill="none" stroke="'+co+'" stroke-width="6" ';
  h+='stroke-dasharray="'+ci2+'" stroke-dashoffset="'+of+'" stroke-linecap="round" style="transition:stroke-dashoffset 1s"/>';
  h+='</svg><div class="v"><span class="n" style="color:'+co+'">'+p+'</span>';
  h+='<span class="u">% CONF</span></div></div>';
  h+='<div class="nfo"><div class="lb">'+(rj?'AUCUNE SALLE RECONNUE':'SALLE IDENTIFIEE')+'</div>';
  h+='<div class="rm">'+b+'</div>';
  h+='<div class="st">marge: <b>'+d.margin.toFixed(4)+'</b>';
  if(d.best_match_image)h+=' &middot; meilleure image: <b>'+d.best_match_image+'</b>';
  h+='</div></div></div>';

  // Scores par salle
  if(sc.length>0){
    h+='<div class="sec"><div class="tt">SCORES PAR SALLE</div>';
    for(const[room,score]of sc){
      const pc=(score/mx*100).toFixed(1);const w=room===d.best_room&&!rj;const cl=gc(room);
      h+='<div class="br'+(w?' w':'')+'"><div class="d" style="background:'+cl+'"></div>';
      h+='<div class="nm">'+room+(w?'<span class="bg">MATCH</span>':'')+'</div>';
      h+='<div class="tk"><div class="fl" style="width:'+pc+'%;background:'+cl+'"></div></div>';
      h+='<div class="sc" style="color:'+(w?cl:'#8b949e')+'">'+score.toFixed(4)+'</div></div>';
    }
    h+='</div>';
  }

  // Top images
  if(matches.length>0){
    h+='<div class="sec"><div class="tt">TOP IMAGES LES PLUS PROCHES</div>';
    for(let i=0;i<Math.min(matches.length,8);i++){
      const m=matches[i];const cl=gc(m.room);
      h+='<div class="mr"><span class="rk">#'+(i+1)+'</span>';
      h+='<div class="dt" style="background:'+cl+'"></div>';
      h+='<span class="inf"><strong>'+m.room+'</strong> &middot; '+m.image+'</span>';
      h+='<span class="sm" style="color:'+cl+'">'+m.similarity.toFixed(4)+'</span></div>';
    }
    h+='</div>';
  }

  // Rejet
  if(rj){
    h+='<div class="rjb">Cette photo ne correspond a aucune salle connue.<br>';
    h+='<span style="color:#8b949e">Score '+c.toFixed(4)+' &lt; seuil '+d.rejection_threshold.toFixed(2)+'</span></div>';
  }

  h+='</div>';
  document.getElementById('out').innerHTML=h;
}
</script></body></html>"""


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

def cmd_index(args):
    idx = RoomIndex()
    idx.build(args.rooms_dir, backend=args.backend, max_images_per_room=args.max_images)
    idx.save(args.output)


def cmd_query(args):
    idx = RoomIndex.load(args.index)
    print(f"  📊 {len(idx.rooms)} salles, {len(idx.metadata)} images, backend={idx.backend}")
    print(f"  📷 Query: {args.image}")

    t0 = time.time()
    r = idx.query(args.image, top_k=args.top_k)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  RÉSULTAT ({elapsed:.2f}s)")
    print(f"{'='*60}")

    if r["rejected"]:
        print(f"\n  🚫 REJETÉ — Salle inconnue")
        print(f"  📊 Score: {r['confidence']:.4f} < seuil {r['rejection_threshold']:.4f}")
    else:
        print(f"\n  🏆 Salle: {r['best_room']}")
        print(f"  📊 Confiance: {r['confidence']:.4f}")
        print(f"  📏 Marge: {r['margin']:.4f}")
        print(f"  🖼️  Image la plus proche: {r['best_match_image']}")

    print(f"\n  Scores par salle:")
    for room, sc in sorted(r['room_scores'].items(), key=lambda x: -x[1]):
        bar = "█" * int(max(0, sc) * 30)
        m = " ◀" if room == r.get('best_room') and not r['rejected'] else ""
        print(f"    {room:25s} {sc:.4f} {bar}{m}")

    if r['top_matches']:
        print(f"\n  Top {min(5, len(r['top_matches']))} images:")
        for i, m in enumerate(r['top_matches'][:5]):
            print(f"    {i+1}. [{m['room']}] {m['image']} (sim={m['similarity']:.4f})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 JSON: {args.output_json}")


def cmd_serve(args):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import tempfile
    import cgi

    idx = RoomIndex.load(args.index)
    print(f"  📊 {len(idx.rooms)} salles, {len(idx.metadata)} images, backend={idx.backend}")
    print(f"  🚫 Seuil: {idx.rejection_threshold:.4f}")

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/localize":
                ct = self.headers.get("Content-Type", "")
                if "multipart" in ct:
                    form = cgi.FieldStorage(
                        fp=self.rfile, headers=self.headers,
                        environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ct})
                    fi = form["image"]
                    if fi.file:
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
                            t.write(fi.file.read()); tp = t.name
                        r = idx.query(tp)
                        os.unlink(tp)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(json.dumps(r).encode())
                        return
                self.send_response(400)
                self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "rooms": idx.rooms,
                    "total_images": len(idx.metadata),
                    "backend": idx.backend,
                    "threshold": idx.rejection_threshold,
                }).encode())
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(WEB_UI.encode())

        def log_message(self, *a):
            pass

    print(f"\n  🌐 http://localhost:{args.port}")
    print(f"  Ouvre cette URL dans ton navigateur")
    HTTPServer(("0.0.0.0", args.port), H).serve_forever()


def main():
    p = argparse.ArgumentParser(description="Room Localization — Image Retrieval (Methode 1)")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("index", help="Indexer les images des salles")
    s.add_argument("--rooms-dir", required=True, help="Dossier avec sous-dossiers par salle")
    s.add_argument("--output", default="index_images.pkl")
    s.add_argument("--backend", default="auto", choices=["auto", "clip", "efficientnet_b0", "resnet50", "resnet18"])
    s.add_argument("--max-images", type=int, default=500, help="Max images par salle")

    s = sub.add_parser("query", help="Localiser une photo")
    s.add_argument("--index", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--output-json")

    s = sub.add_parser("serve", help="Serveur web")
    s.add_argument("--index", required=True)
    s.add_argument("--port", type=int, default=8000)

    args = p.parse_args()
    cmds = {"index": cmd_index, "query": cmd_query, "serve": cmd_serve}
    cmds.get(args.cmd, lambda a: p.print_help())(args)


if __name__ == "__main__":
    main()
