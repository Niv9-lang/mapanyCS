# Application de Navigation 3D MapAnything

Guide complet pour générer une reconstruction 3D et la naviguer interactivement.

## Flux Complet

```
Images
   ↓
[run_reconstruction.py] 
   ↓
Fichier PLY
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
cd /mapanyCS/navigation

# Créer l'environnement (si pas déjà fait)
conda create -n mapanything python=3.12 -y
conda activate mapanything

# Installer MapAnything depuis la racine (si pas déjà fait)
cd ..
pip install -e .


# Installer les dépendances web
pip install flask trimesh

# Ou pour GPU NVIDIA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour Apple Silicon:
pip install torch torchvision torchaudio
```

### Générer la reconstruction

```bash

python reconstruction_medium.py \
    --image_folder ../sol_vivant \
    --output sol_vivant.ply
```

### Résultat
Un fichier `sol_vivant.ply` est créé

---

## Étape 2 : Navigation Interactve

### Installation des dépendances web

```bash
# Installer Flask et trimesh
pip install flask trimesh
```

Pour naviguer dans le modèle 3D:
```bash

python Navigation/navigation_obstacle.py --model salle_vivant_2.ply --ref-dir ../img_mapanything3
# le ref dir permet de choisir un dossier de référence pour la fonction de GPS (déterminer sa position à partir d'une image dans la salle)

python Navigation/navigation_obstacle.py --model salle_vivant_2.ply --ref-dir ../4E_CS
```
Le paramètre --model permet de spécifier le fichier .ply à utiliser et le paramètre --ref-dir permet de spécifier le dossier d'images à utiliser pour établir la localisation à partir d'une photo/

Les fonctionnalités sont visibles aux 3 adresses suivantes (en local) :

http://127.0.0.1:5000/ : Navigation dans le modèle 3D avec la souris

http://127.0.0.1:5000/minimap-distance : mesure de la distance entre deux points dans le modèle 3D

http://127.0.0.1:5000/pathfinding : trajectoires entre deux points en tenant compte des obstacles


## Étape 3 : Annotation des Salles de Classe

Cette fonctionnalité permet d'annoter des salles sur la carte 3D (minimap) issu du fichier PLY_explorer.html, d'afficher une bannière de proximité et de se téléporter instantanément dans une salle via une barre de recherche.

### Configurer `salles.json`

Créez ou éditez le fichier `salles.json` à la racine du projet (`mapanyCS/salles.json`).
Ce fichier doit être chargé depuis le PLY_explorer.html pour pouvoir profiter de l'annotation des salles. Il suffit de press Echap et de charger `salles.json` depuis votre ordinateur.

```json
{
  "proximite_seuil": 2.0,
  "salles": [
    {
      "id": "101",
      "nom": "Salle 101",
      "x": 1.23,
      "y": 0.0,
      "z": -2.45,
      "description": "Salle de cours - Rez-de-chaussée"
    },
    {
      "id": "102",
      "nom": "Salle 102",
      "x": 3.10,
      "y": 0.0,
      "z": -2.45,
      "description": "Laboratoire"
    }
  ]
}
```
# Benchmark

Pour lancer un benchmark avec un dossier à spécifier :

```
python benchmark.py --image_folder img_mapanything/
```
Métriques mesurées :

Temps total (secondes)
RAM pic & moyenne (Go)
CPU pic & moyenne (%)
GPU mémoire pic (Go) — NVIDIA uniquement via pynvml
Nombre de points reconstruits (lu dans l'en-tête PLY)

Les données sont enregistrés dans un fichier json nommé all_results.json	

Pour comparer les performances :

```
python benchmark.py --compare
```