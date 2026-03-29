#!/usr/bin/env python3
"""
Application de Navigation 3D pour Modèles GLB et PLY
Permet de naviguer et explorer les reconstructions 3D avec repérage spatial.
Supporte les maillages GLB et les nuages de points PLY.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from flask import Flask, render_template, send_file, jsonify, request

try:
    import trimesh
except ImportError:
    print("⚠️  trimesh non installé. Installez avec: pip install trimesh")
    trimesh = None


class ModelNavigator:
    """Gestionnaire de navigation pour fichiers GLB ou PLY"""

    def __init__(self, model_path: str):
        """
        Initialiser le navigateur.
        
        Args:
            model_path: Chemin vers le fichier GLB ou PLY
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {model_path}")

        self.suffix = self.model_path.suffix.lower()
        if self.suffix not in ('.glb', '.ply'):
            raise ValueError(f"Format non supporté: {self.suffix}. Utilisez .glb ou .ply")

        self.geometry = None  # Trimesh ou PointCloud
        self.vertices = None  # numpy array des sommets
        self.bounds = None
        self.center = None
        self.radius = None
        self.metadata = {}
        
        self._load_model()
        self._compute_bounds()

    def _load_model(self):
        """Charger le modèle GLB ou PLY. Récupération des distances"""
        if trimesh is None:
            raise ImportError("trimesh requis pour le chargement")
        
        try:
            loaded = trimesh.load(str(self.model_path))
            
            if self.suffix == '.glb':
                if isinstance(loaded, trimesh.Scene):
                    meshes = [
                        geom for geom in loaded.geometry.values()
                        if isinstance(geom, trimesh.Trimesh)
                    ]
                    if not meshes:
                        raise ValueError("Aucun maillage trouvé dans la scène GLB")
                    self.geometry = trimesh.util.concatenate(meshes)
                else:
                    self.geometry = loaded
                self.vertices = np.asarray(self.geometry.vertices)
                n_faces = len(self.geometry.faces) if hasattr(self.geometry, 'faces') else 0
                print(f"✓ GLB chargé: {self.model_path.name}")
                print(f"  Sommets: {len(self.vertices):,} | Faces: {n_faces:,}")
            else:
                # PLY - nuage de points
                if isinstance(loaded, trimesh.PointCloud):
                    self.geometry = loaded
                    self.vertices = np.asarray(loaded.vertices)
                elif isinstance(loaded, trimesh.Trimesh):
                    self.geometry = loaded
                    self.vertices = np.asarray(loaded.vertices)
                else:
                    self.vertices = np.asarray(loaded.vertices) if hasattr(loaded, 'vertices') else np.array(loaded)
                    self.geometry = loaded
                print(f"✓ PLY chargé: {self.model_path.name}")
                print(f"  Points: {len(self.vertices):,}")
        except Exception as e:
            raise RuntimeError(f"Erreur chargement {self.suffix}: {e}")

    def _compute_bounds(self):
        """Calculer les limites du modèle"""
        if self.vertices is not None and len(self.vertices) > 0:
            vertices = self.vertices
            self.bounds = {
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            }
            self.center = ((np.array(self.bounds["min"]) + np.array(self.bounds["max"])) / 2).tolist()
            self.radius = float(np.linalg.norm(
                np.array(self.bounds["max"]) - np.array(self.bounds["min"])
            ) / 2)
            
            print(f"✓ Limites calculées:")
            print(f"  Centre: {self.center}")
            print(f"  Rayon: {self.radius:.2f}")

    def get_scene_info(self) -> Dict:
        """Obtenir les informations de la scène pour le frontend"""
        return {
            "bounds": self.bounds,
            "center": self.center,
            "radius": self.radius,
            "filename": self.model_path.name,
            "model_format": self.suffix[1:],  # "glb" ou "ply"
            "model_url": "/model",
        }

    def get_camera_config(self) -> Dict:
        """Obtenir la configuration de caméra recommandée"""
        if self.center is None or self.radius is None:
            return {"position": [0, 0, 10], "fov": 75, "near": 0.1, "far": 10000}
        
        center = np.array(self.center)
        # Positionner la caméra pour voir l'objet entièrement
        camera_dist = self.radius * 2.5
        
        return {
            "position": (center + np.array([0, self.radius * 0.5, camera_dist])).tolist(),
            "target": self.center,
            "fov": 60,
            "near": 0.1,
            "far": self.radius * 100,
        }

    def compute_minimap_data(self) -> Dict:
        """Calculer les données pour la minimap"""
        if self.vertices is None or len(self.vertices) == 0:
            return {}
        
        vertices = self.vertices
        
        # Échantillonner les vertices pour le dessin (max 3000 points)
        n = len(vertices)
        step = max(1, n // 3000)
        sample = vertices[::step]
        xz_points = sample[:, [0, 2]].tolist()  # Projection XZ (vue de dessus)
        xy_points = sample[:, [0, 1]].tolist()  # Projection XY (vue de face)
        yz_points = sample[:, [1, 2]].tolist()  # Projection YZ (vue de côté)
        
        return {
            "bounds_xz": {
                "min": [float(vertices[:, 0].min()), float(vertices[:, 2].min())],
                "max": [float(vertices[:, 0].max()), float(vertices[:, 2].max())],
            },
            "bounds_xy": {
                "min": [float(vertices[:, 0].min()), float(vertices[:, 1].min())],
                "max": [float(vertices[:, 0].max()), float(vertices[:, 1].max())],
            },
            "bounds_yz": {
                "min": [float(vertices[:, 1].min()), float(vertices[:, 2].min())],
                "max": [float(vertices[:, 1].max()), float(vertices[:, 2].max())],
            },
            "vertices_xz": xz_points,
            "vertices_xy": xy_points,
            "vertices_yz": yz_points,
        }


def create_app(model_path: str) -> Flask:
    """
    Créer l'application Flask
    
    Args:
        model_path: Chemin vers le fichier GLB ou PLY
        
    Returns:
        Application Flask configurée
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # Initialiser le navigateur
    try:
        navigator = ModelNavigator(model_path)
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None

    # ==================== ROUTES ====================

    @app.route("/")
    def index():
        """Serve the main page"""
        return render_template("index.html")

    @app.route("/minimap-distance")
    def minimap_distance():
        """Page dédiée à la minimap + mesure de distances"""
        return render_template("minimap_distance.html")

    @app.route("/api/scene-info")
    def get_scene_info():
        """Get scene information"""
        return jsonify(navigator.get_scene_info())

    @app.route("/api/camera-config")
    def get_camera_config():
        """Get recommended camera configuration"""
        return jsonify(navigator.get_camera_config())

    @app.route("/api/minimap-data")
    def get_minimap_data():
        """Get minimap data"""
        return jsonify(navigator.compute_minimap_data())

    @app.route("/model")
    @app.route("/model.glb")
    @app.route("/model.ply")
    def get_model():
        """Sert le modèle GLB ou PLY"""
        try:
            path = navigator.model_path.resolve()
            if not path.exists():
                return f"Fichier non trouvé: {path}", 404
            mimetype = "model/gltf-binary" if navigator.suffix == ".glb" else "application/octet-stream"
            return send_file(
                str(path),
                mimetype=mimetype,
                as_attachment=False,
            )
        except Exception as e:
            print(f"✗ Erreur envoi modèle: {e}")
            return str(e), 500

    @app.route("/api/help")
    def get_help():
        """Get navigation help"""
        return jsonify({
            "controls": {
                "rotate": "Bouton droit + souris / Deux doigts",
                "pan": "Bouton central + souris / Ctrl + Bouton droit",
                "zoom": "Molette de souris / Pincement",
                "reset": "Touche R",
                "toggle_minimap": "Touche M",
                "screenshot": "Touche P",
            },
            "stats": navigator.get_scene_info(),
        })

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Application de Navigation 3D pour Modèles GLB ou PLY"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Chemin vers le fichier GLB ou PLY",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Adresse de l'hôte (défaut: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port (défaut: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mode debug",
    )

    args = parser.parse_args()

    # Créer l'application
    print("=" * 60)
    print("🚀 Démarrage de l'Application de Navigation 3D")
    print("=" * 60)

    app = create_app(args.model)
    if app is None:
        return 1

    # Vérifier les fichiers template
    template_path = Path("templates")
    static_path = Path("static")
    
    if not template_path.exists():
        print(f"\n⚠️  Dossier 'templates' non trouvé")
        print("   Les fichiers seront créés automatiquement...")

    print(f"\n✓ Application prête!")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"\n  Appuyez sur Ctrl+C pour arrêter")
    print("=" * 60)

    # Lancer le serveur
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    return 0


if __name__ == "__main__":
    exit(main())
