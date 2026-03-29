# 🗺️ Application de Navigation 3D MapAnything

Guide complet pour générer une reconstruction 3D et la naviguer interactivement.

## 📋 Flux Complet

```
Images
   ↓
[run_reconstruction.py] 
   ↓
Fichier GLB 3D
   ↓
[navigation.py]
   ↓
Application Web Interactive
```

---

## Étape 1 : Génération de la Reconstruction 3D

### Formats supportés
- **GLB** : maillage 3D (généré par `run_reconstruction.py`)
- **PLY** : nuage de points (généré par `reconstruction_medium.py`) — meilleure qualité, idéal pour calcul de distances

### Installation des dépendances

```bash
# Naviguer vers le dossier navigation
cd Desktop/map-anything/navigation

# Créer l'environnement (si pas déjà fait)
conda create -n mapanything python=3.12 -y
conda activate mapanything

# Installer MapAnything depuis la racine (si pas déjà fait)
cd ..
pip install -e .
cd navigation

# Installer les dépendances web
pip install flask trimesh

# Ou pour GPU NVIDIA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour Apple Silicon:
pip install torch torchvision torchaudio
```

### Générer la reconstruction

```bash
# Depuis le dossier navigation/ avec images relatives
python reconstruction_medium.py \
    --image_folder ../sol_vivant \
    --output sol_vivant.ply

# Ou avec chemin absolu
python reconstruction_medium.py \
    --image_folder /chemin/vers/vos/images \
    --output_glb resultat.ply
```

**Options disponibles:**
```bash
python run_reconstruction.py --help

--image_folder      Dossier contenant les images (REQUIS)
--output_glb        Fichier de sortie (défaut: reconstruction.glb)
--model {apache,cc-by-nc}  Modèle à utiliser (défaut: apache)
--device {cuda,cpu} Processeur (défaut: auto)
```

### Résultat
Un fichier `reconstruction.glb` ou `salle_LC.ply` est créé

**Alternative PLY (nuage de points, meilleure qualité) :**
```bash
python reconstruction_medium.py --image_folder ../img_mapanything --output ma_salle.ply
```

---

## Étape 2 : Navigation Interactve

### Installation des dépendances web

```bash
# Installer Flask et trimesh
pip install flask trimesh
```

### Lancer l'application

**Option A : Utiliser le script bash (recommandé)**
```bash
cd map-anything/navigation
chmod +x run_navigation.sh
./run_navigation.sh resultat.glb
```

**Option B : Commande Python directe**
```bash
cd map-anything/Navigation
python navigation.py --model ../salle_vivant_mast3r.ply
# ou pour un fichier PLY (nuage de points)
python navigation.py --model ../sol_vivant.ply

python navigation_obstacle.py --model ../sol_vivant.ply --host 127.0.0.1 --port 5000


#######################################################################################
python reconstruction_medium.py --image_folder ../img_mapanything3 --output salle_vivant_2.ply

python Navigation/navigation_obstacle.py --model salle_vivant_2.ply --ref-dir ../img_mapanything3




