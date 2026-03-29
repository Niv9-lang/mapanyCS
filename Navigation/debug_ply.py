import numpy as np
import trimesh

pcd = trimesh.load("../salle_vivant.ply")
y = np.asarray(pcd.vertices)[:, 1]

print(f"Y min : {y.min():.4f}")
print(f"Y max : {y.max():.4f}")
print(f"Y médian : {np.median(y):.4f}")

# Histogramme simplifié
hist, bins = np.histogram(y, bins=50)
peak_idx = hist.argmax()
print(f"Pic dominant Y ≈ {(bins[peak_idx]+bins[peak_idx+1])/2:.4f}  (candidat sol)")