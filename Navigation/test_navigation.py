#!/usr/bin/env python3
"""
Test rapide de l'application de navigation 3D
Vérifie les dépendances et la configuration
"""

import sys
import subprocess
from pathlib import Path

def test_import(package_name, display_name=None):
    """Tester l'import d'un package"""
    if display_name is None:
        display_name = package_name
    
    try:
        __import__(package_name)
        print(f"✓ {display_name}")
        return True
    except ImportError:
        print(f"✗ {display_name} - MANQUANT")
        return False

def test_structure():
    """Vérifier la structure du projet"""
    print("\n📁 Structure du projet:")
    
    required_files = [
        ("navigation.py", "Script principal"),
        ("templates/index.html", "Page HTML"),
        ("static/js/navigation.js", "Logic JavaScript"),
        ("static/css/style.css", "Styles CSS"),
        ("run_reconstruction.py", "Script de reconstruction"),
    ]
    
    all_ok = True
    for file_path, description in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {description}: {file_path}")
        else:
            print(f"✗ {description}: {file_path} - MANQUANT")
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("🔍 Test de l'Application de Navigation 3D")
    print("="*60)

    # 1. Vérifier la structure
    print("\n🔧 Vérification des fichiers...")
    structure_ok = test_structure()

    # 2. Vérifier les dépendances Python
    print("\n📦 Vérification des dépendances Python:")
    
    deps_ok = True
    deps_ok &= test_import("flask", "Flask (serveur web)")
    deps_ok &= test_import("mapanything", "MapAnything")
    deps_ok &= test_import("torch", "PyTorch")
    deps_ok &= test_import("numpy", "NumPy")
    
    # Optionnels pour la navigation
    has_trimesh = test_import("trimesh", "Trimesh (GLB loading)")
    
    print("\n⚙️ Dépendances optionnelles:")
    test_import("cv2", "OpenCV")
    test_import("PIL", "Pillow")

    # 3. Vérifier le GPU
    print("\n💻 Détection GPU:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("✗ Pas de GPU CUDA - Utilisation de CPU")
    except:
        print("✗ PyTorch non disponible")

    # 4. Summary
    print("\n" + "="*60)
    if structure_ok and deps_ok and has_trimesh:
        print("✓ Configuration OK - Vous êtes prêt à commencer!")
        print("\nPour démarrer:")
        print("  1. python run_reconstruction.py --image_folder ~/images --output_glb model.glb")
        print("  2. python navigation.py --model model.glb")
        print("  3. Ouvrir http://127.0.0.1:5000")
        return 0
    else:
        print("⚠️  Problèmes détectés")
        print("\nPour installer les dépendances manquantes:")
        print("  pip install flask trimesh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
