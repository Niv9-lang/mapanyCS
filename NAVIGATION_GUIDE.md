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

python reconstruction_medium.py \
    --image_folder ../sol_vivant \
    --output sol_vivant.ply
```

### Résultat
Un fichier `salle_LC.ply` est créé

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
# le ref dir permet de choisir un dossier de référence pour la fonction de GPS (déterminer sa position à partir d'une image dans la salle)

python Navigation/navigation_obstacle.py --model salle_vivant_2.ply --ref-dir ../4E_CS
```

---

## Étape 2 : Liste des fonctionnalités disponibles

http://127.0.0.1:5000/ : Navigation dans le modèle 3D avec la souris

http://127.0.0.1:5000/minimap-distance : mesure de la distance entre deux points dans le modèle 3D

http://127.0.0.1:5000/pathfinding : trajectoires entre deux points en tenant compte des obstacles


## Étape 3 : Annotation des Salles de Classe

Cette fonctionnalité permet d'annoter des salles sur la carte 2D (minimap), d'afficher
une bannière de proximité et de se téléporter instantanément dans une salle via une barre
de recherche.

### Configurer `salles.json`

Créez ou éditez le fichier `salles.json` à la racine du projet (`map-anything/salles.json`) :

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

**Champs :**
- `proximite_seuil` : distance (en unités 3D) à partir de laquelle la bannière de proximité s'affiche
- `id` : identifiant court affiché sur la minimap (ex. `"101"`)
- `nom` : nom complet affiché dans la barre de recherche et la bannière
- `x`, `y`, `z` : coordonnées 3D de la salle — lire les valeurs affichées dans le panneau **Ma position** de l'explorateur en naviguant jusqu'à la porte de chaque salle
- `description` : texte libre optionnel (affiché dans les résultats de recherche et la bannière)

### Trouver les coordonnées d'une salle

1. Lancer le serveur et ouvrir `http://127.0.0.1:5000`
2. Naviguer dans la vue 3D jusqu'à la porte de la salle souhaitée
3. Lire les coordonnées X / Y / Z dans le panneau **Ma position** (coin supérieur gauche)
4. Copier ces valeurs dans `salles.json`
5. Recharger la page (`F5`) — les annotations apparaissent immédiatement

### Lancer le serveur avec les salles

# Combiné avec la localisation visuelle
```
python Navigation/navigation_obstacle.py --model salle_LC.ply \
    --ref-dir ../salle_directory \
    --salles salles.json
```

### Fonctionnalités disponibles

| Fonctionnalité | Description |
|---|---|
| **Étiquettes minimap** | Chaque salle affichée en bleu sur la carte 2D ; devient orange à proximité |
| **Bannière de proximité** | S'affiche en bas de l'écran quand la caméra est à ≤ `proximite_seuil` unités d'une salle |
| **Barre de recherche** | En haut de l'écran — taper un numéro ou un nom, cliquer pour se téléporter |
| **Touche `S`** | Raccourci clavier pour activer la barre de recherche |
| **Téléportation** | La caméra se déplace instantanément devant la salle sélectionnée |
