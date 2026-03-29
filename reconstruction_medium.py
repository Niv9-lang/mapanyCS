#!/usr/bin/env python3
"""
Reconstruction 3D - Point cloud PLY à partir d'images
Utilise MapAnything pour générer une reconstruction 3D colorée (nuage de points PLY).
"""

import os
import sys
import glob
import json
import argparse
import torch
import numpy as np
import open3d as o3d

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.models import init_model_from_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")

# 2. Loading the MapAnything Model from HuggingFace Hub

model = MapAnything.from_pretrained("facebook/map-anything").to(device)

# model = init_model_from_config("mast3r", device="cuda")
#ou
#model = MapAnything.from_pretrained("facebook/map-anything-mast3r").to(device)

# For Apache 2.0 license model, use "facebook/map-anything-apache"

# 3. Import images (chemin par défaut ou argument)
parser = argparse.ArgumentParser(description="Reconstruction 3D en nuage de points PLY")
parser.add_argument("--image_folder", type=str, default="img_mapanything",
                    help="Dossier contenant les images")
parser.add_argument("--output", type=str, default="salle_LC.ply",
                    help="Fichier PLY de sortie")
parser.add_argument("--no_visualize", action="store_true",
                    help="Ne pas ouvrir la fenêtre Open3D de visualisation")
args = parser.parse_args()

set_of_images = args.image_folder
if not os.path.isdir(set_of_images):
    print(f"✗ Dossier introuvable: {set_of_images}")
    print("  Placez vos images dans un dossier (ex: img_mapanything/) ou utilisez --image_folder")
    sys.exit(1)

views = load_images(set_of_images)
print(f"✓ {len(views)} images chargées depuis {set_of_images}")
if len(views) == 0:
    print("✗ Aucune image trouvée!")
    sys.exit(1) 


# 4. Running Inference with Optimized Parameters
predictions = model.infer(
    views,                            # Input views
    memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
    use_amp=True,                     # Use mixed precision inference (recommended)
    amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
    apply_mask=True,                  # Apply masking to dense geometry outputs
    mask_edges=True,                  # Remove edge artifacts by using normals and depth
    apply_confidence_mask=False,      # Filter low-confidence regions
    confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
)

# 5. Extracting Valid 3D Points with Mask Filtering

def extract_points_from_prediction(prediction, apply_mask=True):
    """Extract valid 3D points and mask from a single view prediction.
       La variable pts3d contiendra les coordonnées métriques fournies 
       par le modèle choisi (mapanything, mast3r)

       Elles sont ensuite sauvegardés dans le fichier .ply (voir section #9)
    """
    pts3d = prediction["pts3d"].cpu().numpy()
    
    if pts3d.ndim == 4:
        pts3d = pts3d[0]
    
    if apply_mask and "mask" in prediction:
        mask = prediction["mask"].cpu().numpy()
        if mask.ndim == 4:
            mask = mask[0, :, :, 0]
        elif mask.ndim == 3:
            mask = mask[0]
        mask_bool = mask > 0.5
    else:
        mask_bool = np.ones((pts3d.shape[0], pts3d.shape[1]), dtype=bool)
    
    points = pts3d[mask_bool]
    return points, mask_bool



# 8. Mapping RGB Colors from Source Images
def extract_colors_from_prediction(prediction, mask_indices):
    """Extract RGB colors for valid points from the input image."""
    img = prediction["img_no_norm"].cpu().numpy()
    
    if img.ndim == 4:
        img = img[0]
    
    colors = img[mask_indices]
    colors = np.clip(colors, 0.0, 1.0)
    return colors

# 9. Merging Multi-View Predictions into Complete Reconstruction
def merge_all_views_to_pointcloud(predictions, apply_mask=True, verbose=True):
    """Merge all view predictions into a single Open3D point cloud with RGB colors."""
    
    all_points = []
    all_colors = []
    
    for i, pred in enumerate(predictions):
        points, mask = extract_points_from_prediction(pred, apply_mask=apply_mask)
        colors = extract_colors_from_prediction(pred, mask)
        
        all_points.append(points)
        all_colors.append(colors)
        
        if verbose:
            print(f"View {i+1}/{len(predictions)}: {points.shape[0]:,} points")
    
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    if verbose:
        print(f"\n✅ Total merged points: {merged_points.shape[0]:,}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    
    return pcd


pcd_complete = merge_all_views_to_pointcloud(predictions, apply_mask=True, verbose=True)

# Visualize complete reconstruction (sauf si --no_visualize)
if not args.no_visualize:
    o3d.visualization.draw_geometries([pcd_complete], window_name="Complete 3D Reconstruction")

o3d.io.write_point_cloud(args.output, pcd_complete)
print(f"\n✓ Reconstruction sauvegardée: {args.output}")

# ── Sauvegarde des poses pour la localisation visuelle ──────────────
# Récupère les fichiers images dans le même ordre que load_images (tri alpha)
img_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
img_files = sorted(f for ext in img_exts for f in glob.glob(os.path.join(set_of_images, ext)))

poses = {}
for i, img_path in enumerate(img_files):
    if i >= len(predictions):
        break
    pred = predictions[i]
    fname = os.path.basename(img_path)

    if "camera_poses" in pred:
        # Matrice 4×4 caméra→monde : la translation [:3, 3] = position de la caméra
        pose = pred["camera_poses"]
        if hasattr(pose, "cpu"):
            pose = pose.cpu().numpy()
        if pose.ndim == 3:
            pose = pose[0]          # enlever la dimension batch
        cam_pos = pose[:3, 3].tolist()
    else:
        # Repli : centroïde des points 3D de cette vue
        pts, _ = extract_points_from_prediction(pred, apply_mask=True)
        cam_pos = pts.mean(axis=0).tolist() if len(pts) > 0 else [0.0, 0.0, 0.0]

    poses[fname] = cam_pos

poses_path = os.path.join(set_of_images, "poses.json")
with open(poses_path, "w") as f:
    json.dump(poses, f, indent=2)
print(f"✓ Poses sauvegardées : {poses_path}  ({len(poses)} images)")
print(f"\n  → Pour la localisation, lancer le serveur avec :")
print(f"     python Navigation/navigation_obstacle.py --model {args.output} --ref-dir {set_of_images}")