#!/usr/bin/env python3
"""
MapAnything — Distances & Trajectoire A→B
Supporte GLB et PLY.
"""

import heapq
import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, render_template, send_file, jsonify, request

try:
    import trimesh
except ImportError:
    print("trimesh non installé : pip install trimesh")
    trimesh = None


class ModelNavigator:

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {model_path}")
        self.suffix = self.model_path.suffix.lower()
        if self.suffix not in ('.glb', '.ply'):
            raise ValueError(f"Format non supporté : {self.suffix}")

        self.geometry = None
        self.vertices = None
        self.bounds   = None
        self.center   = None
        self.radius   = None
        self._cached_grid        = None
        self._cached_grid_params = None

        self._load_model()
        self._compute_bounds()
        self._analyze_y()

    # ──────────────────────────────────────────────
    #  Chargement
    # ──────────────────────────────────────────────

    def _load_model(self):
        if trimesh is None:
            raise ImportError("trimesh requis")
        loaded = trimesh.load(str(self.model_path))
        if self.suffix == '.glb':
            if isinstance(loaded, trimesh.Scene):
                meshes = [g for g in loaded.geometry.values()
                          if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    raise ValueError("Aucun maillage GLB")
                self.geometry = trimesh.util.concatenate(meshes)
            else:
                self.geometry = loaded
            self.vertices = np.asarray(self.geometry.vertices, dtype=np.float64)
            print(f"✓ GLB : {len(self.vertices):,} sommets")
        else:
            self.geometry = loaded
            self.vertices = np.asarray(
                loaded.vertices if hasattr(loaded, 'vertices') else loaded,
                dtype=np.float64
            )
            print(f"✓ PLY : {len(self.vertices):,} points")

    def _compute_bounds(self):
        v = self.vertices
        self.bounds = {"min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist()}
        self.center = ((v.min(axis=0) + v.max(axis=0)) / 2).tolist()
        self.radius = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)) / 2)
        sz = v.max(axis=0) - v.min(axis=0)
        self.explorer_scale = float(30.0 / (sz.max() or 1.0))

    def get_explorer_transform(self) -> Dict:
        """
        Retourne les paramètres de la transformation appliquée par PLY_explorer.html :
          1. Centrage sur la bounding box
          2. Normalisation à 30 unités (scale = 30 / maxD)
          3. Rotation selon l'axe vertical (worldUpPreset)
        Permet de convertir les coordonnées brutes PLY en coordonnées viewer PLY_explorer.
        """
        horiz = [i for i in range(3) if i != self.vertical_axis]
        # Conventions de signe issues des cas worldUpPreset de PLY_explorer.html :
        #   vertical_axis=0 (X-up, preset 4) : x_sign=-1, z_sign=+1
        #   vertical_axis=1 (Y-up, preset 0) : x_sign=+1, z_sign=+1
        #   vertical_axis=2 (Z-up, preset 2) : x_sign=+1, z_sign=-1
        x_sign = -1 if self.vertical_axis == 0 else 1
        z_sign = -1 if self.vertical_axis == 2 else 1
        return {
            "scale":         self.explorer_scale,
            "center":        self.center,        # [ctr_x, ctr_y, ctr_z] coordonnées PLY brutes
            "x_center_col":  horiz[0],           # index dans center[] pour l'axe X curseur
            "z_center_col":  horiz[1],           # index dans center[] pour l'axe Z curseur
            "x_sign":        x_sign,
            "z_sign":        z_sign,
        }

    def floor_y_from_viewer(self, viewer_y: float, up_preset: int) -> float:
        """
        Convertit une coordonnée Y viewer (Three.js, telle que stockée dans floorY du JSON)
        en coordonnée brute PLY sur l'axe vertical détecté.

        Formule inverse de la transformation PLY_explorer :
          viewer_Y = vert_sign * (PLY_vert - center_vert) * explorer_scale
          → PLY_vert = vert_sign * viewer_Y / explorer_scale + center_vert

        Signe selon upPreset (même logique que les rotations switch() de PLY_explorer) :
          preset pair  (0,2,4) → vert_sign = +1
          preset impair(1,3,5) → vert_sign = -1
        """
        vert_sign = 1 if up_preset % 2 == 0 else -1
        va = self.vertical_axis
        return vert_sign * viewer_y / self.explorer_scale + self.center[va]

    def apply_floor_from_json(self, floor_y_viewer: float, up_preset: int) -> None:
        """Applique un floorY issu du JSON (coordonnées viewer) sur l'axe vertical PLY brut."""
        ply_floor = self.floor_y_from_viewer(floor_y_viewer, up_preset)
        self.floor_y_center = ply_floor
        cluster_half = self.y_range * 0.05
        y = self.vertices[:, self.vertical_axis]
        floor_pts = y[(y >= ply_floor - cluster_half) & (y <= ply_floor + cluster_half)]
        if len(floor_pts) >= 10:
            self.floor_y_min = float(floor_pts.min())
            self.floor_y_max = float(floor_pts.max())
        else:
            self.floor_y_min = ply_floor - cluster_half
            self.floor_y_max = ply_floor + cluster_half
        self._cached_grid = None  # invalider le cache
        print(f"  floorY JSON appliqué : viewer={floor_y_viewer:.3f} → PLY={ply_floor:.4f}")

    def _detect_vertical_axis(self) -> int:
        """
        Détecte l'axe vertical (0=X, 1=Y, 2=Z) en cherchant celui dont
        l'histogramme présente les deux pics de densité les plus marqués.
        Sol et plafond sont les deux surfaces les plus denses d'un espace
        intérieur — leur axe commun est l'axe vertical.
        Gère les PLY CloudCompare (Z-up) et MASt3r/Gaussian Splatting (Y-up).
        """
        best_axis, best_score = 1, -1.0
        for ax in range(3):
            vals = self.vertices[:, ax]
            vrange = float(vals.max() - vals.min())
            if vrange < 1e-6:
                continue
            hist, _ = np.histogram(vals, bins=200)
            k  = np.ones(5) / 5
            hs = np.convolve(hist.astype(float), k, mode='same')
            total = hs.sum() or 1.0
            peaks = [i for i in range(1, len(hs) - 1)
                     if hs[i] > hs[i - 1] and hs[i] > hs[i + 1]]
            if len(peaks) < 2:
                continue
            peaks_sorted = sorted(peaks, key=lambda i: hs[i], reverse=True)
            score = (hs[peaks_sorted[0]] + hs[peaks_sorted[1]]) / total
            if score > best_score:
                best_score = score
                best_axis  = ax
        return best_axis

    def _analyze_y(self):
        """
        Détecte l'axe vertical automatiquement puis identifie sol et plafond
        par les deux pics de densité dominants dans l'histogramme de cet axe.
        Le pic le plus bas = sol, le plus haut = plafond.
        Compatible Y-up (MASt3r) et Z-up (CloudCompare).
        """
        self.vertical_axis = self._detect_vertical_axis()
        ax_name = ['X', 'Y', 'Z'][self.vertical_axis]
        print(f"  Axe vertical détecté : {ax_name} (colonne {self.vertical_axis})")

        y = self.vertices[:, self.vertical_axis]
        ymin, ymax = float(y.min()), float(y.max())
        y_range = ymax - ymin

        # Histogramme fin + lissage fenêtre 7
        hist, bins = np.histogram(y, bins=300)
        centers = (bins[:-1] + bins[1:]) / 2
        k = np.ones(7) / 7
        hist_s = np.convolve(hist.astype(float), k, mode='same')

        # ── Deux pics dominants : le plus bas = sol, le plus haut = plafond ──
        peaks = [i for i in range(1, len(hist_s) - 1)
                 if hist_s[i] > hist_s[i - 1] and hist_s[i] > hist_s[i + 1]]
        peaks_sorted = sorted(peaks, key=lambda i: hist_s[i], reverse=True)
        top2 = sorted(peaks_sorted[:2])  # trier par valeur de l'axe croissante

        if len(top2) >= 2:
            floor_y_center = float(centers[top2[0]])
            ceil_y_center  = float(centers[top2[1]])
        elif len(top2) == 1:
            floor_y_center = float(centers[top2[0]])
            ceil_y_center  = ymax
        else:
            floor_y_center = ymin
            ceil_y_center  = ymax

        # ── Bande sol : largeur du pic (±3 % de y_range) ──
        # 3 % est suffisant pour un sol plat ; un sol légèrement incliné
        # reste capturé car on prend le min/max des points dans la bande.
        cluster_half = y_range * 0.05
        floor_pts = y[(y >= floor_y_center - cluster_half) & (y <= floor_y_center + cluster_half)]
        if len(floor_pts) >= 10:
            self.floor_y_min = float(floor_pts.min())
            self.floor_y_max = float(floor_pts.max())
        else:
            self.floor_y_min = floor_y_center - cluster_half
            self.floor_y_max = floor_y_center + cluster_half

        self.floor_y_center = floor_y_center
        self.ceil_y_center  = ceil_y_center
        self.y_range        = y_range

        n_floor = int(((y >= self.floor_y_min) & (y <= self.floor_y_max)).sum())
        print(f"  {ax_name} brut PLY : min={ymin:.4f}  max={ymax:.4f}  étendue={y_range:.4f}")
        print(f"  Sol détecté : centre={floor_y_center:.4f}  "
              f"bande=[{self.floor_y_min:.4f}, {self.floor_y_max:.4f}]  "
              f"points sol={n_floor:,}")
        print(f"  Plafond : centre={ceil_y_center:.4f}")

    # ──────────────────────────────────────────────
    #  Infos scène / caméra / minimap
    # ──────────────────────────────────────────────

    def get_scene_info(self) -> Dict:
        return {
            "bounds": self.bounds, "center": self.center, "radius": self.radius,
            "filename": self.model_path.name,
        }

    def compute_minimap_data(self) -> Dict:
        v = self.vertices
        step = max(1, len(v) // 3000)
        s = v[::step]
        return {
            "bounds_xz": {"min": [float(v[:,0].min()), float(v[:,2].min())],
                          "max": [float(v[:,0].max()), float(v[:,2].max())]},
            "bounds_xy": {"min": [float(v[:,0].min()), float(v[:,1].min())],
                          "max": [float(v[:,0].max()), float(v[:,1].max())]},
            "bounds_yz": {"min": [float(v[:,1].min()), float(v[:,2].min())],
                          "max": [float(v[:,1].max()), float(v[:,2].max())]},
            "vertices_xz": s[:, [0, 2]].tolist(),
            "vertices_xy": s[:, [0, 1]].tolist(),
            "vertices_yz": s[:, [1, 2]].tolist(),
        }

    def get_floor_info(self) -> Dict:
        """Retourne les infos sur le sol détecté pour le frontend."""
        y = self.vertices[:, self.vertical_axis]
        y_ply_min    = float(y.min())
        y_ply_max    = float(y.max())
        y_ply_center = (y_ply_min + y_ply_max) / 2.0
        # Conversion vers coordonnées viewer Three.js :
        #   Y_viewer = Y_ply_center - Y_ply
        # Attention : l'ordre min/max s'inverse à cause de la négation.
        floor_min_v = round(y_ply_center - self.floor_y_max, 6)
        floor_max_v = round(y_ply_center - self.floor_y_min, 6)
        return {
            "y_min":               y_ply_min,
            "y_max":               y_ply_max,
            "y_range":             float(self.y_range),
            "y_ply_center":        y_ply_center,
            "floor_y_center":      float(self.floor_y_center),
            "floor_y_min":         float(self.floor_y_min),
            "floor_y_max":         float(self.floor_y_max),
            "floor_y_min_viewer":  floor_min_v,
            "floor_y_max_viewer":  floor_max_v,
            "ceil_y_center":       float(self.ceil_y_center) if self.ceil_y_center else None,
            "vertical_axis":       ['X', 'Y', 'Z'][self.vertical_axis],
        }

    # ──────────────────────────────────────────────
    #  GRILLE D'OCCUPATION
    #
    #  Algorithme :
    #  1. Identifier les points du SOL (Y dans la bande floor_y ± tolerance)
    #  2. Identifier les OBSTACLES (points à hauteur h_min..h_max au-dessus du sol)
    #     "au-dessus" dépend de l'orientation détectée
    #  3. Construire la grille :
    #     - Cellule avec point sol ET sans obstacle → LIBRE (0)
    #     - Cellule avec obstacle → OBSTACLE (1)
    #     - Sinon → INCONNU (2)
    #  4. Flood-fill depuis le centroïde des cellules libres pour
    #     marquer l'intérieur de la salle comme LIBRE même sans point
    # ──────────────────────────────────────────────

    def compute_occupancy_grid(
        self,
        grid_size: int          = 512,
        robot_radius_cells: int = 2,
        floor_band_override: Optional[Tuple[float, float]] = None,
        obs_min_count: int      = 5,      # nb minimum de points pour marquer une cellule obstacle
        floor_tolerance: float  = 0.20,   # largeur totale de la bande sol (en unités PLY, généralement mètres)
    ) -> Dict:

        if self.vertices is None or len(self.vertices) < 100:
            return {"error": "Pas assez de points"}

        params_key = (grid_size, robot_radius_cells, floor_band_override, obs_min_count, floor_tolerance)
        if self._cached_grid is not None and self._cached_grid_params == params_key:
            return self._cached_grid

        v = self.vertices
        # Axes horizontaux = les deux axes qui ne sont pas l'axe vertical
        horiz = [i for i in range(3) if i != self.vertical_axis]
        x, y, z = v[:, horiz[0]], v[:, self.vertical_axis], v[:, horiz[1]]

        xmin, xmax = float(x.min()), float(x.max())
        zmin, zmax = float(z.min()), float(z.max())
        x_range = xmax - xmin or 1e-6
        z_range = zmax - zmin or 1e-6

        gs = grid_size

        # ── Référence sol globale (même logique que PLY_explorer.html buildOccGrid) ──
        floor_center = self.floor_y_center
        band_half    = floor_tolerance / 2.0
        room_height  = abs(self.ceil_y_center - self.floor_y_center)
        obstacle_max_h = room_height if room_height > band_half * 2 else self.y_range * 0.80

        # Sens vertical : les obstacles sont-ils au-dessus ou en-dessous du sol ?
        above = int((y > floor_center + band_half).sum())
        below = int((y < floor_center - band_half).sum())
        obstacles_above_floor = above >= below

        print(f"  Sol centre={floor_center:.4f}  band_half={band_half:.3f}  "
              f"obs_max_h={obstacle_max_h:.3f}  sens={'↑' if obstacles_above_floor else '↓'}")

        # ── Indices de cellule ──
        cols = np.clip(((x - xmin) / x_range * (gs - 1)).astype(int), 0, gs - 1)
        rows = np.clip(((z - zmin) / z_range * (gs - 1)).astype(int), 0, gs - 1)

        # ── Classification identique à PLY_explorer buildOccGrid ──
        # h_above = hauteur au-dessus du sol global (référence unique, pas locale)
        if obstacles_above_floor:
            h_above = y - floor_center
        else:
            h_above = floor_center - y

        floor_mask = (h_above >= -band_half) & (h_above <= band_half)
        obs_mask   = (h_above >  band_half)  & (h_above <  obstacle_max_h)

        print(f"  Points sol={floor_mask.sum():,}  obstacles={obs_mask.sum():,}")

        # ── Grille sol / comptage obstacles ──
        has_floor = np.zeros((gs, gs), dtype=bool)
        obs_count = np.zeros((gs, gs), dtype=np.int32)

        has_floor[rows[floor_mask], cols[floor_mask]] = True
        np.add.at(obs_count, (rows[obs_mask], cols[obs_mask]), 1)
        has_obs = obs_count >= obs_min_count

        # ── Grille ternaire initiale ──
        # 0 = libre, 1 = obstacle, 2 = inconnu
        grid = np.full((gs, gs), 2, dtype=np.uint8)
        grid[has_floor & ~has_obs] = 0   # sol sans obstacle → libre
        grid[has_obs]              = 1   # obstacle

        # ── Flood-fill : propager LIBRE depuis les cellules sol vers l'intérieur ──
        # Les cellules intérieures sans point (inconnu) deviennent libres si elles
        # sont accessibles depuis le sol sans traverser un obstacle.
        grid = self._flood_fill_free(grid, gs)

        # ── Dilatation des obstacles pour marge robot ──
        if robot_radius_cells > 0:
            grid = self._dilate_obstacles(grid, robot_radius_cells)

        n_free = int((grid == 0).sum())
        n_obs  = int((grid == 1).sum())
        n_unk  = int((grid == 2).sum())
        pct    = 100.0 * n_obs / (gs * gs)

        print(f"  Grille {gs}×{gs} : libre={n_free:,}  obstacle={n_obs:,} ({pct:.1f}%)  "
              f"inconnu={n_unk:,}")

        result = {
            "grid":             grid.tolist(),
            "grid_size":        gs,
            "bounds":           {"xmin": xmin, "xmax": xmax, "zmin": zmin, "zmax": zmax},
            "floor_y_min":      float(floor_center - band_half),
            "floor_y_max":      float(floor_center + band_half),
            "obstacles_above":  bool(obstacles_above_floor),
            "floor_tolerance":  float(floor_tolerance),
            "obstacle_max_h":   float(obstacle_max_h),
            "room_height":      float(room_height),
            "obstacle_cells":   n_obs,
            "free_cells":       n_free,
            "unknown_cells":    n_unk,
            "total_cells":      gs * gs,
            "robot_radius_cells": robot_radius_cells,
        }
        self._cached_grid        = result
        self._cached_grid_params = params_key
        return result

    # ──────────────────────────────────────────────
    #  Helpers grille
    # ──────────────────────────────────────────────

    def _flood_fill_free(self, grid: np.ndarray, gs: int) -> np.ndarray:
        """
        Propagation BFS depuis toutes les cellules libres (sol détecté).
        Les cellules INCONNUES adjacentes à des cellules libres deviennent libres,
        sauf si elles sont séparées par un obstacle.
        Cela remplit l'intérieur de la salle même sans points de scan.
        """
        result = grid.copy()
        # Graines : toutes les cellules libres
        seeds = deque(zip(*np.where(result == 0)))
        visited = (result == 0)

        while seeds:
            r, c = seeds.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < gs and 0 <= nc < gs and not visited[nr, nc]:
                    visited[nr, nc] = True
                    if result[nr, nc] == 2:   # inconnu → libre
                        result[nr, nc] = 0
                        seeds.append((nr, nc))
                    # Si obstacle (1) : ne pas traverser, ne pas propager

        return result

    def _dilate_obstacles(self, grid: np.ndarray, radius: int) -> np.ndarray:
        """Dilate les obstacles vers les cellules libres adjacentes."""
        result    = grid.copy()
        rs, cs    = grid.shape
        positions = np.argwhere(grid == 1)
        if len(positions) == 0:
            return result
        d = radius
        dy, dx = np.mgrid[-d:d+1, -d:d+1]
        circle = (dy**2 + dx**2) <= radius**2
        cy_o, cx_o = np.where(circle)
        cy_o -= d; cx_o -= d
        for r, c in positions:
            nr = r + cy_o; nc = c + cx_o
            ok = (nr >= 0) & (nr < rs) & (nc >= 0) & (nc < cs)
            tr = nr[ok]; tc = nc[ok]
            free_mask = result[tr, tc] == 0
            result[tr[free_mask], tc[free_mask]] = 1
        return result

    def _compute_local_floor_heightmap(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        y: np.ndarray,
        gs: int,
        obstacles_above: bool,
        fy_min: float,
        fy_max: float,
    ) -> np.ndarray:
        """
        Carte de hauteur locale du sol, cellule par cellule.

        Pour chaque cellule de la grille on cherche la hauteur du sol
        local parmi les points tombant dans une bande de recherche centrée
        sur la bande sol [fy_min, fy_max].

        La bande de recherche = max(3 × largeur bande sol, 2 % de y_range).
        Cela la rend sensible aux corrections manuelles (override), tout en
        restant assez large pour couvrir les sols en pente.

        Paramètres
        ----------
        rows, cols : indices de cellule pour chaque point (déjà clippés)
        y          : coordonnée Y brute de chaque point
        gs         : taille de la grille
        obstacles_above : True si les obstacles ont Y > sol, False si Y < sol
        fy_min, fy_max  : bande sol utilisée (auto-détectée ou override)

        Retourne
        --------
        np.ndarray de forme (gs, gs), float64.
        NaN sur les cellules sans aucun point dans la bande de recherche
        (comblé ensuite par interpolation par voisinage).
        """
        fy_center   = (fy_min + fy_max) / 2
        band_width  = max(fy_max - fy_min, 1e-6)
        y_range     = float(y.max() - y.min()) or 1.0
        # Bande de recherche : assez large pour les sols en pente,
        # mais proportionnelle à la bande sol pour que l'override ait un effet.
        search_half = max(band_width * 3.0, y_range * 0.02)
        in_band = (y >= fy_center - search_half) & (y <= fy_center + search_half)

        rs = rows[in_band]
        cs = cols[in_band]
        ys = y[in_band]

        if len(ys) == 0:
            # Repli : utiliser tous les points
            rs, cs, ys = rows, cols, y

        cell_idx = rs * gs + cs
        n = gs * gs

        if obstacles_above:
            # Sol = minimum Y local dans la bande
            fmap = np.full(n, np.inf, dtype=np.float64)
            np.minimum.at(fmap, cell_idx, ys)
            fmap[fmap == np.inf] = np.nan
        else:
            # Sol = maximum Y local (orientation inversée)
            fmap = np.full(n, -np.inf, dtype=np.float64)
            np.maximum.at(fmap, cell_idx, ys)
            fmap[fmap == -np.inf] = np.nan

        fmap = fmap.reshape(gs, gs)
        fmap = self._fill_nan_heightmap(fmap, fy_center)
        return fmap

    def _fill_nan_heightmap(
        self, hmap: np.ndarray, fallback: float, max_passes: int = 20
    ) -> np.ndarray:
        """
        Comble les cellules NaN de la carte de hauteur par moyenne
        itérative des 4-voisins valides.
        Les cellules encore NaN après max_passes reçoivent la valeur
        de repli (centre du sol global).
        """
        result = hmap.copy()
        for _ in range(max_passes):
            nans = np.isnan(result)
            if not nans.any():
                break
            pad  = np.pad(result, 1, constant_values=np.nan)
            nbrs = np.stack(
                [pad[:-2, 1:-1], pad[2:, 1:-1],
                 pad[1:-1, :-2], pad[1:-1, 2:]],
                axis=-1,
            )
            avg    = np.nanmean(nbrs, axis=-1)
            filled = ~np.isnan(avg)
            result = np.where(nans & filled, avg, result)
        # Valeur de repli pour les rares cellules encore NaN
        result = np.where(np.isnan(result), fallback, result)
        return result

    def _nearest_free(self, grid: np.ndarray, pos: Tuple, gs: int) -> Optional[Tuple]:
        q = deque([pos]); seen = {pos}
        while q:
            r, c = q.popleft()
            if grid[r, c] == 0:
                return (r, c)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < gs and 0 <= nc < gs and (nr,nc) not in seen:
                    if abs(nr-pos[0])+abs(nc-pos[1]) <= 80:
                        seen.add((nr,nc)); q.append((nr,nc))
        return None

    # ──────────────────────────────────────────────
    #  PLANIFICATION A*
    # ──────────────────────────────────────────────

    def find_path(
        self,
        ax: float, az: float,
        bx: float, bz: float,
        grid_size: int = 512,
        robot_radius_cells: int = 2,
        floor_band_override: Optional[Tuple[float, float]] = None,
        obs_min_count: int = 5,
        floor_tolerance: float = 0.20,
        precomputed_grid: Optional[Dict] = None,
    ) -> Dict:

        # ── Grille d'affichage ──
        # Si une grille uploadée depuis PLY_explorer est disponible, l'utiliser directement.
        # Sinon, calculer en Python (fallback).
        if precomputed_grid:
            gd = precomputed_grid
        else:
            gd = self.compute_occupancy_grid(
                grid_size, robot_radius_cells, floor_band_override, obs_min_count, floor_tolerance)
            if "error" in gd:
                return {"error": gd["error"]}

        # ── Grille de planification (même source, re-dilater) ──
        planning_radius = robot_radius_cells * 2
        if precomputed_grid:
            # Repartir de la grille uploadée et dilater davantage pour la planification
            gd_plan = precomputed_grid
        else:
            gd_plan = self.compute_occupancy_grid(
                grid_size, planning_radius, floor_band_override, obs_min_count, floor_tolerance)

        grid = np.array(gd_plan["grid"], dtype=np.uint8)
        gs   = gd["grid_size"]
        b    = gd["bounds"]          # même bounds pour les deux grilles
        xr   = b["xmax"] - b["xmin"] or 1e-6
        zr   = b["zmax"] - b["zmin"] or 1e-6

        def w2g(wx, wz):
            c = int(np.clip((wx - b["xmin"]) / xr * (gs-1), 0, gs-1))
            r = int(np.clip((wz - b["zmin"]) / zr * (gs-1), 0, gs-1))
            return r, c

        def g2w(r, c):
            return b["xmin"] + c/(gs-1)*xr, b["zmin"] + r/(gs-1)*zr

        start = w2g(ax, az)
        end   = w2g(bx, bz)

        # Snapper sur la grille de PLANIFICATION (marge 2×), pas d'affichage
        if grid[start] != 0:
            start = self._nearest_free(grid, start, gs) or start
        if grid[end] != 0:
            end = self._nearest_free(grid, end, gs) or end

        print(f"  A* : {start} → {end}")

        dirs = [(-1,0,1.),(1,0,1.),(0,-1,1.),(0,1,1.),
                (-1,-1,1.414),(-1,1,1.414),(1,-1,1.414),(1,1,1.414)]
        heur = lambda a, b: ((a[0]-b[0])**2 + (a[1]-b[1])**2)**.5

        heap = [(heur(start, end), 0., start)]
        came: Dict = {}
        g_sc: Dict = {start: 0.}
        closed: set = set()
        found = False; iters = 0
        # Limite = toutes les cellules libres de la grille (borne supérieure réaliste)
        max_iters = int((grid == 0).sum()) + 1

        while heap and iters < max_iters:
            _, g, cur = heapq.heappop(heap)
            if cur in closed: continue        # pop obsolète, ne compte pas
            if cur == end: found = True; break
            closed.add(cur)
            iters += 1                        # compter uniquement les vraies expansions
            for dr, dc, cost in dirs:
                nr, nc = cur[0]+dr, cur[1]+dc
                if not (0 <= nr < gs and 0 <= nc < gs): continue
                if (nr,nc) in closed or grid[nr,nc] != 0: continue
                ng = g_sc[cur] + cost
                if (nr,nc) not in g_sc or ng < g_sc[(nr,nc)]:
                    g_sc[(nr,nc)] = ng
                    heapq.heappush(heap, (ng + heur((nr,nc), end), ng, (nr,nc)))
                    came[(nr,nc)] = cur

        if not found:
            return {"error": "Aucun chemin trouvé — les points sont peut-être dans une zone obstacle ou inconnue",
                    "grid_data": gd, "iterations": iters}

        path_g: List[Tuple] = []
        cur = end
        while cur in came: path_g.append(cur); cur = came[cur]
        path_g.append(start); path_g.reverse()

        smooth = self._smooth_path(path_g, grid, gs)
        path_w = [g2w(r, c) for r, c in smooth]
        length = sum(
            ((path_w[i][0]-path_w[i-1][0])**2 + (path_w[i][1]-path_w[i-1][1])**2)**.5
            for i in range(1, len(path_w))
        )

        return {
            "path":             [[float(p[0]), float(p[1])] for p in path_w],
            "path_raw_count":   len(path_g),
            "path_smooth_count": len(smooth),
            "length":           float(length),
            "iterations":       iters,
            "grid_data":        gd,
        }

    def _smooth_path(self, path, grid, gs):
        if len(path) <= 2: return path
        smoothed = [path[0]]; i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i+1:
                if self._los(grid, path[i], path[j]): break
                j -= 1
            smoothed.append(path[j]); i = j
        return smoothed

    def _los(self, grid, a, b):
        r0, c0 = a; r1, c1 = b
        dr, dc = abs(r1-r0), abs(c1-c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        rs, cs = grid.shape
        while True:
            if not (0 <= r0 < rs and 0 <= c0 < cs): return False
            if grid[r0, c0] != 0: return False
            if r0 == r1 and c0 == c1: return True
            e2 = 2*err
            if e2 > -dc: err -= dc; r0 += sr
            if e2 <  dr: err += dr; c0 += sc


# ══════════════════════════════════════════════════════════════════
#  Flask App
# ══════════════════════════════════════════════════════════════════

def create_app(model_path: str, salles_path: Optional[str] = None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

    try:
        nav = ModelNavigator(model_path)
    except Exception as e:
        print(f"✗ {e}"); return None

    # Fichier JSON des salles : ordre de priorité si --salles non fourni :
    #   1. {nom_modele}.json à côté du PLY  (ex: vivant_RDC.json)
    #   2. salles.json à côté du PLY
    #   3. {nom_modele}.json à la racine du projet
    #   4. salles.json à la racine du projet
    if salles_path:
        _salles_path = Path(salles_path)
    else:
        model_p   = Path(model_path)
        stem_json = model_p.parent / (model_p.stem + ".json")
        sibling   = model_p.parent / "salles.json"
        root_stem = Path(model_p.stem + ".json")
        root_def  = Path("salles.json")
        _salles_path = (stem_json  if stem_json.exists()  else
                        sibling    if sibling.exists()    else
                        root_stem  if root_stem.exists()  else
                        root_def)
    if _salles_path.exists():
        print(f"  Salles JSON       : {_salles_path}")
        import json as _json
        try:
            with open(_salles_path, encoding="utf-8") as _f:
                _jd = _json.load(_f)
            _floor_y = _jd.get("floorY") or _jd.get("floor")
            _up_preset = int(_jd.get("upPreset", 1))
            if isinstance(_floor_y, (int, float)):
                nav.apply_floor_from_json(float(_floor_y), _up_preset)
        except Exception as _e:
            print(f"  Avertissement JSON : {_e}")
    else:
        print(f"  Salles JSON       : introuvable ({_salles_path})")

    @app.route("/")
    def index(): return render_template("pathfinding.html")

    @app.route("/api/salles")
    def get_salles():
        if _salles_path.exists():
            import json as _json
            with open(_salles_path, encoding="utf-8") as f:
                return jsonify(_json.load(f))
        return jsonify({"proximite_seuil": 2.0, "salles": []})

    @app.route("/minimap-distance")
    def minimap_distance(): return render_template("minimap_distance.html")

    @app.route("/api/scene-info")
    def scene_info(): return jsonify(nav.get_scene_info())

    @app.route("/api/minimap-data")
    def minimap_data(): return jsonify(nav.compute_minimap_data())

    # Grille uploadée depuis PLY_explorer.html (via _uploadGridToFlask)
    _grid_store: Dict = {}

    @app.route("/api/grid-upload", methods=["POST", "OPTIONS"])
    def grid_upload():
        # CORS : autoriser PLY_explorer (port 8080) à poster vers Flask (port 5000)
        if request.method == "OPTIONS":
            from flask import Response as _R
            r = _R()
            r.headers["Access-Control-Allow-Origin"]  = "*"
            r.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
            r.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return r
        import base64 as _b64
        data = request.json
        gs   = int(data["grid_size"])
        raw  = np.frombuffer(_b64.b64decode(data["grid"]), dtype=np.uint8).reshape(gs, gs)
        n_free = int((raw == 0).sum())
        n_obs  = int((raw == 1).sum())
        _grid_store["grid"] = {
            "grid":            raw.tolist(),
            "grid_size":       gs,
            "bounds":          {"xmin": data["xmin"], "xmax": data["xmax"],
                                "zmin": data["zmin"], "zmax": data["zmax"]},
            "floor_y_min":     float(data.get("floor_y_center", 0)) - 0.10,
            "floor_y_max":     float(data.get("floor_y_center", 0)) + 0.10,
            "obstacles_above": True,
            "floor_tolerance": 0.20,
            "obstacle_cells":  n_obs,
            "free_cells":      n_free,
            "unknown_cells":   int((raw == 2).sum()),
            "total_cells":     gs * gs,
            "robot_radius_cells": 2,
        }
        print(f"  ✓ Grille reçue de PLY_explorer : {gs}×{gs}  libre={n_free:,}  obstacle={n_obs:,}")
        r = jsonify({"ok": True})
        r.headers["Access-Control-Allow-Origin"] = "*"
        return r

    @app.route("/pathfinding")
    def pathfinding_page(): return render_template("pathfinding.html")

    @app.route("/api/floor-info")
    def floor_info():
        return jsonify(nav.get_floor_info())

    @app.route("/api/explorer-transform")
    def explorer_transform():
        return jsonify(nav.get_explorer_transform())

    @app.route("/api/occupancy-grid")
    def get_grid():
        # Priorité : grille uploadée depuis PLY_explorer (garantie correcte)
        if _grid_store.get("grid"):
            return jsonify(_grid_store["grid"])
        # Fallback : calcul Python (si PLY_explorer pas ouvert)
        gs        = 512
        obs_min   = 5
        floor_tol = max(0.01, float(request.args.get("floor_tolerance", 0.20)))
        return jsonify(nav.compute_occupancy_grid(gs, 2, None, obs_min, floor_tol))

    @app.route("/api/pathfind")
    def pathfind():
        try:
            ax=float(request.args["ax"]); az=float(request.args["az"])
            bx=float(request.args["bx"]); bz=float(request.args["bz"])
        except (KeyError, ValueError) as e:
            return jsonify({"error": f"ax,az,bx,bz requis : {e}"}), 400
        floor_tol = max(0.01, float(request.args.get("floor_tolerance", 0.20)))
        print(f"\n🗺  A→B : ({ax:.3f},{az:.3f}) → ({bx:.3f},{bz:.3f})")
        # Utiliser la grille uploadée depuis PLY_explorer si disponible
        stored = _grid_store.get("grid")
        return jsonify(nav.find_path(ax, az, bx, bz,
                                     floor_tolerance=floor_tol,
                                     precomputed_grid=stored))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Chemin vers le fichier PLY ou GLB")
    parser.add_argument("--salles", default=None,  help="Fichier JSON des salles (défaut: <modèle>.json)")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    print("="*60 + "\n🚀  MapAnything — Distances & Trajectoire\n" + "="*60)
    app = create_app(args.model, args.salles)
    if app is None: return 1

    print(f"\n  Distances       → http://{args.host}:{args.port}/minimap-distance")
    print(f"  Trajectoire A→B → http://{args.host}:{args.port}/pathfinding\n" + "="*60)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    exit(main())
