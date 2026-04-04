#!/usr/bin/env python3
"""
Application de Navigation 3D — MapAnything
Supporte GLB et PLY. Inclut planification de trajectoire A*.
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

try:
    from localization import VisualLocalizer
except ImportError:
    VisualLocalizer = None
    print("localization.py introuvable")


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

    def _analyze_y(self):
        """
        Détecte le sol automatiquement par analyse des extrêmes Y.

        Principe : les photos étant prises par un humain debout, MASt3r place
        toujours le sol et le plafond aux abscisses Y les plus extrêmes du nuage.
        Les 4 murs créent des pics intermédiaires dans l'histogramme Y.
        On cherche donc le pic dominant dans les 15 % inférieurs et supérieurs de
        l'étendue Y, et on prend le plus dense comme sol.

        L'étendue réelle du sol (pente incluse) est déterminée en isolant
        le cluster de points autour du centre détecté (±10 % de y_range).

        Coordonnées PLY brutes (avant rotation Three.js).
        Le viewer applique rotation.x = PI + recentrage :
            Y_viewer = Y_ply_center - Y_ply
        """
        y = self.vertices[:, 1]
        ymin, ymax = float(y.min()), float(y.max())
        y_range = ymax - ymin

        # Histogramme fin + lissage
        hist, bins = np.histogram(y, bins=300)
        centers = (bins[:-1] + bins[1:]) / 2
        k = np.ones(7) / 7
        hist_s = np.convolve(hist.astype(float), k, mode='same')

        # ── Pic dominant dans les 15 % inférieurs (candidat sol ou plafond bas) ──
        margin = y_range * 0.15
        mask_bot = centers <= ymin + margin
        mask_top = centers >= ymax - margin

        if mask_bot.any():
            idx_bot      = int(np.argmax(hist_s[mask_bot]))
            bot_center   = float(centers[mask_bot][idx_bot])
            bot_density  = float(hist_s[mask_bot][idx_bot])
        else:
            bot_center, bot_density = ymin, 0.0

        if mask_top.any():
            idx_top      = int(np.argmax(hist_s[mask_top]))
            top_center   = float(centers[mask_top][idx_top])
            top_density  = float(hist_s[mask_top][idx_top])
        else:
            top_center, top_density = ymax, 0.0

        # ── Le sol est l'extrême le plus dense ──
        if bot_density >= top_density:
            floor_y_center = bot_center
            ceil_y_center  = top_center
        else:
            floor_y_center = top_center
            ceil_y_center  = bot_center

        # ── Étendue réelle du sol : cluster ±10 % de y_range autour du centre ──
        # Cela capture la pente du sol sans fixer une bande arbitraire.
        cluster_half = y_range * 0.10
        floor_pts = y[(y >= floor_y_center - cluster_half) & (y <= floor_y_center + cluster_half)]
        if len(floor_pts) >= 10:
            self.floor_y_min = float(floor_pts.min())
            self.floor_y_max = float(floor_pts.max())
        else:
            self.floor_y_min = floor_y_center - y_range * 0.05
            self.floor_y_max = floor_y_center + y_range * 0.05

        self.floor_y_center = floor_y_center
        self.ceil_y_center  = ceil_y_center
        self.y_range        = y_range

        n_floor = int(((y >= self.floor_y_min) & (y <= self.floor_y_max)).sum())
        print(f"  Y brut PLY : min={ymin:.4f}  max={ymax:.4f}  étendue={y_range:.4f}")
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
            "model_format": self.suffix[1:], "model_url": "/model",
        }

    def get_camera_config(self) -> Dict:
        if self.center is None:
            return {"position": [0, 0, 10], "fov": 75, "near": 0.1, "far": 10000}
        c = np.array(self.center)
        return {
            "position": (c + np.array([0, self.radius * 0.5, self.radius * 2.5])).tolist(),
            "target": self.center, "fov": 60, "near": 0.1, "far": self.radius * 100,
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
        y = self.vertices[:, 1]
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
            # Coordonnées telles que vues dans le viewer (rotation.x=PI + centrage)
            "floor_y_min_viewer":  floor_min_v,
            "floor_y_max_viewer":  floor_max_v,
            "ceil_y_center":       float(self.ceil_y_center) if self.ceil_y_center else None,
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
        grid_size: int          = 256,
        obstacle_min_h: float   = 0.10,   # hauteur min (au-dessus du sol) pour être obstacle
        obstacle_max_h: float   = 2.00,   # hauteur max obstacle (au-delà = plafond, ignoré)
        robot_radius_cells: int = 2,
        floor_band_override: Optional[Tuple[float, float]] = None,
    ) -> Dict:

        if self.vertices is None or len(self.vertices) < 100:
            return {"error": "Pas assez de points"}

        params_key = (grid_size, obstacle_min_h, obstacle_max_h,
                      robot_radius_cells, floor_band_override)
        if self._cached_grid is not None and self._cached_grid_params == params_key:
            return self._cached_grid

        v = self.vertices
        x, y, z = v[:, 0], v[:, 1], v[:, 2]

        xmin, xmax = float(x.min()), float(x.max())
        zmin, zmax = float(z.min()), float(z.max())
        x_range = xmax - xmin or 1e-6
        z_range = zmax - zmin or 1e-6

        gs = grid_size

        # ── Bande sol ──
        if floor_band_override:
            fy_min, fy_max = floor_band_override
        else:
            fy_min, fy_max = self.floor_y_min, self.floor_y_max

        fy_center = (fy_min + fy_max) / 2

        # ── Orientation : le sol est-il au-dessous ou au-dessus en Y brut ? ──
        # On regarde si le reste des points est majoritairement au-dessus ou en-dessous du sol.
        non_floor = y[(y < fy_min) | (y > fy_max)]
        if len(non_floor) > 0:
            above = int((non_floor > fy_center).sum())
            below = int((non_floor < fy_center).sum())
            # Les murs + plafond sont du côté où il y a le plus de points
            obstacles_above_floor = above >= below
        else:
            obstacles_above_floor = True

        print(f"  Sol : [{fy_min:.4f}, {fy_max:.4f}] | "
              f"obstacles {'au-dessus' if obstacles_above_floor else 'en-dessous'} du sol")

        # ── Indices de cellule (tous les points) ──
        cols = np.clip(((x - xmin) / x_range * (gs - 1)).astype(int), 0, gs - 1)
        rows = np.clip(((z - zmin) / z_range * (gs - 1)).astype(int), 0, gs - 1)

        # ── Carte de hauteur locale du sol (gestion des sols en pente) ──
        #
        #  Pour chaque cellule XZ de la grille, on calcule la hauteur du sol
        #  local à partir des points dans une bande centrée sur [fy_min, fy_max].
        #  La bande est proportionnelle à la largeur de la bande sol, ce qui la
        #  rend sensible aux corrections manuelles tout en couvrant les pentes.
        local_floor = self._compute_local_floor_heightmap(
            rows, cols, y, gs,
            obstacles_above_floor, fy_min, fy_max,
        )

        # ── Hauteur de chaque point au-dessus du sol local ──
        local_fy_at_pt  = local_floor[rows, cols]
        valid_floor_map = ~np.isnan(local_fy_at_pt)

        # Tolérance verticale : demi-largeur de la bande sol globale détectée
        band_half = (fy_max - fy_min) / 2

        if obstacles_above_floor:
            # Y croît vers le haut
            h_above = y - local_fy_at_pt
        else:
            # Y décroît vers le haut (orientation inversée)
            h_above = local_fy_at_pt - y

        # Sol : point dans ±band_half du plancher local
        floor_mask = valid_floor_map & (h_above >= -band_half) & (h_above <= band_half)
        # Obstacle : hauteur au-dessus du sol comprise dans [min_h, max_h]
        obs_mask   = valid_floor_map & (h_above > obstacle_min_h) & (h_above < obstacle_max_h)

        print(f"  Points sol={floor_mask.sum():,}  obstacles={obs_mask.sum():,}")

        # ── Grille booléenne sol / obstacle ──
        has_floor = np.zeros((gs, gs), dtype=bool)
        has_obs   = np.zeros((gs, gs), dtype=bool)

        has_floor[rows[floor_mask], cols[floor_mask]] = True
        has_obs  [rows[obs_mask],   cols[obs_mask]]   = True

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
            "floor_y_min":      float(fy_min),
            "floor_y_max":      float(fy_max),
            "obstacles_above":  bool(obstacles_above_floor),
            "obstacle_min_h":   float(obstacle_min_h),
            "obstacle_max_h":   float(obstacle_max_h),
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
        grid_size: int = 256,
        obstacle_min_h: float = 0.10,
        obstacle_max_h: float = 2.00,
        robot_radius_cells: int = 2,
        floor_band_override: Optional[Tuple[float, float]] = None,
    ) -> Dict:

        # ── Grille d'affichage (marge = 1× rayon robot) ──
        gd = self.compute_occupancy_grid(
            grid_size, obstacle_min_h, obstacle_max_h,
            robot_radius_cells, floor_band_override)
        if "error" in gd:
            return {"error": gd["error"]}

        # ── Grille de planification (marge = 2× rayon robot) ──
        # On dilate le double pour que le chemin garde un espace visible
        # entre sa trajectoire et la bordure des zones rouges affichées.
        # La grille d'affichage reste inchangée : le rouge ne s'élargit pas.
        planning_radius = robot_radius_cells * 2
        gd_plan = self.compute_occupancy_grid(
            grid_size, obstacle_min_h, obstacle_max_h,
            planning_radius, floor_band_override)

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

def create_app(model_path: str, ref_dir: Optional[str] = None, salles_path: Optional[str] = None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

    try:
        nav = ModelNavigator(model_path)
    except Exception as e:
        print(f"✗ {e}"); return None

    # Localisation visuelle (optionnelle, requiert --ref-dir)
    localizer = VisualLocalizer(ref_dir) if VisualLocalizer else None

    # Fichier JSON des salles (cherche salles.json à côté du modèle si non spécifié)
    _salles_path = Path(salles_path) if salles_path else Path(model_path).parent / "salles.json"
    if not _salles_path.exists():
        # Chercher aussi à la racine du projet (dossier courant)
        _salles_path_root = Path("salles.json")
        if _salles_path_root.exists():
            _salles_path = _salles_path_root

    @app.route("/")
    def index(): return render_template("index.html")

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

    @app.route("/api/camera-config")
    def camera_config(): return jsonify(nav.get_camera_config())

    @app.route("/api/minimap-data")
    def minimap_data(): return jsonify(nav.compute_minimap_data())

    @app.route("/model")
    @app.route("/model.glb")
    @app.route("/model.ply")
    def get_model():
        p = nav.model_path.resolve()
        if not p.exists(): return "Fichier non trouvé", 404
        mime = "model/gltf-binary" if nav.suffix == ".glb" else "application/octet-stream"
        return send_file(str(p), mimetype=mime, as_attachment=False)

    @app.route("/pathfinding")
    def pathfinding_page(): return render_template("pathfinding.html")

    @app.route("/api/floor-info")
    def floor_info():
        return jsonify(nav.get_floor_info())

    @app.route("/api/occupancy-grid")
    def get_grid():
        gs       = max(64, min(512, int(request.args.get("grid_size", 256))))
        min_h    = max(0.001, float(request.args.get("min_h", 0.10)))
        max_h    = max(min_h + 0.01, float(request.args.get("max_h", 2.00)))
        radius   = max(0, min(20, int(request.args.get("robot_radius", 2))))
        fy_min_s = request.args.get("floor_y_min", None)
        fy_max_s = request.args.get("floor_y_max", None)
        band     = None
        if fy_min_s and fy_max_s:
            fmin_v, fmax_v = float(fy_min_s), float(fy_max_s)
            if request.args.get("viewer_coords", "0") == "1":
                # Convertir coordonnées viewer → PLY brutes
                # Y_viewer = Y_ply_center - Y_ply  →  Y_ply = Y_ply_center - Y_viewer
                # L'ordre min/max s'inverse à cause de la négation.
                yc   = (nav.vertices[:, 1].min() + nav.vertices[:, 1].max()) / 2.0
                band = (yc - fmax_v, yc - fmin_v)
            else:
                band = (fmin_v, fmax_v)
        return jsonify(nav.compute_occupancy_grid(gs, min_h, max_h, radius, band))

    @app.route("/api/pathfind")
    def pathfind():
        try:
            ax=float(request.args["ax"]); az=float(request.args["az"])
            bx=float(request.args["bx"]); bz=float(request.args["bz"])
        except (KeyError, ValueError) as e:
            return jsonify({"error": f"ax,az,bx,bz requis : {e}"}), 400
        gs       = max(64, min(512, int(request.args.get("grid_size", 256))))
        min_h    = max(0.001, float(request.args.get("min_h", 0.10)))
        max_h    = max(min_h+0.01, float(request.args.get("max_h", 2.00)))
        radius   = max(0, min(20, int(request.args.get("robot_radius", 2))))
        fy_min_s = request.args.get("floor_y_min", None)
        fy_max_s = request.args.get("floor_y_max", None)
        band     = None
        if fy_min_s and fy_max_s:
            fmin_v, fmax_v = float(fy_min_s), float(fy_max_s)
            if request.args.get("viewer_coords", "0") == "1":
                yc   = (nav.vertices[:, 1].min() + nav.vertices[:, 1].max()) / 2.0
                band = (yc - fmax_v, yc - fmin_v)
            else:
                band = (fmin_v, fmax_v)
        print(f"\n🗺  A→B : ({ax:.3f},{az:.3f}) → ({bx:.3f},{bz:.3f})")
        return jsonify(nav.find_path(ax, az, bx, bz, gs, min_h, max_h, radius, band))

    # ── Localisation visuelle ─────────────────────────────────

    @app.route("/orientation")
    def orientation_page():
        return render_template("orientation.html")

    @app.route("/api/localize/status")
    def localize_status():
        if localizer is None:
            return jsonify({"ready": False, "error": "Module de localisation non chargé."})
        return jsonify(localizer.get_status())

    @app.route("/api/localize/image", methods=["POST"])
    def localize_image():
        if localizer is None:
            return jsonify({"error": "Module de localisation non chargé."}), 503
        f = request.files.get("file")
        if f is None:
            return jsonify({"error": "Champ 'file' manquant."}), 400
        result = localizer.localize_image_bytes(f.read())
        return jsonify(result)

    @app.route("/api/localize/video", methods=["POST"])
    def localize_video():
        if localizer is None:
            return jsonify({"error": "Module de localisation non chargé."}), 503
        f = request.files.get("file")
        if f is None:
            return jsonify({"error": "Champ 'file' manquant."}), 400
        sample_fps = max(0.1, min(10.0, float(request.args.get("sample_fps", 1.0))))
        results    = localizer.localize_video_bytes(f.read(), sample_fps=sample_fps)
        return jsonify(results)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True,  help="Chemin vers le fichier PLY ou GLB")
    parser.add_argument("--ref-dir", default=None,   help="Dossier d'images de référence pour la localisation (images.txt ou poses.json)")
    parser.add_argument("--salles",  default=None,   help="Fichier JSON des salles de classe (défaut: salles.json à côté du modèle)")
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--port",    type=int, default=5000)
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    print("="*60 + "\n🚀  Navigation 3D + Planification + Localisation\n" + "="*60)
    app = create_app(args.model, args.ref_dir, args.salles)
    if app is None: return 1

    print(f"\n  Navigation 3D   → http://{args.host}:{args.port}")
    print(f"  Distances       → http://{args.host}:{args.port}/minimap-distance")
    print(f"  Trajectoire A→B → http://{args.host}:{args.port}/pathfinding")
    print(f"  Orientation GPS → http://{args.host}:{args.port}/orientation\n" + "="*60)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    exit(main())
