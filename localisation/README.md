# Localisation par Image + OCR

## Installation

```bash
pip install torch torchvision open_clip_torch numpy pillow
pip install easyocr opencv-python
pip install pytesseract
pip install matplotlib
pip install pillow-heif
pip install flask
```

Pour Tesseract : telecharger le .exe depuis https://github.com/UB-Mannheim/tesseract/wiki

## Indexer les images (une seule fois)

```bash
python localize_images.py index --rooms-dir ./mon_batiment --output index.pkl
```

## Fichiers

| Fichier | Description |
|---|---|
| localize_images.py | Localisation par CLIP seul |
| localize_hybrid_EasyOCR.py | Localisation CLIP + EasyOCR (2 methodes de fusion) |
| localize_hybrid_Tesseract.py | Localisation CLIP + Tesseract (comparaison) |
| optimize_weights.py | Courbes d'optimisation des poids |
| interface.py | Interface web locale |
| labels.json | Labels des salles |

## Interface web (recommande)

Lancer l'interface graphique locale :

```bash
python interface.py --index index.pkl --labels labels.json
```

Ouvrir http://localhost:5000 dans le navigateur. Glisser une image et cliquer sur Methode Lineaire ou Methode Softmax.

Pour changer le port :
```bash
python interface.py --index index.pkl --labels labels.json --port 8080
```

## Commandes terminal

### CLIP seul

```bash
python localize_images.py query --index index.pkl --image photo.png
```

### CLIP + EasyOCR — Methode Lineaire (alpha = 0.7)

Score = 0.7 x CLIP + 0.3 x OCR

```bash
python localize_hybrid_EasyOCR.py query --index index.pkl --labels labels.json --image photo.png --method linear
```

Avec poids personnalises :
```bash
python localize_hybrid_EasyOCR.py query --index index.pkl --labels labels.json --image photo.png --method linear --clip-weight 0.6 --ocr-weight 0.4
```

### CLIP + EasyOCR — Methode Softmax (article)

Normalisation Min-Max + Softmax dynamique (T=0.7)

```bash
python localize_hybrid_EasyOCR.py query --index index.pkl --labels labels.json --image photo.png --method softmax
```

Avec temperature personnalisee :
```bash
python localize_hybrid_EasyOCR.py query --index index.pkl --labels labels.json --image photo.png --method softmax --temperature 0.5
```

### Test OCR seul

```bash
python localize_hybrid_EasyOCR.py ocr-test --image photo.png --labels labels.json
```

### CLIP + Tesseract (comparaison)

```bash
python localize_hybrid_Tesseract.py ocr-test --image photo.png --labels labels.json
python localize_hybrid_Tesseract.py query --index index.pkl --labels labels.json --image photo.png
```

### Optimisation des poids

Image unique :
```bash
python optimize_weights.py --index index.pkl --labels labels.json --image photo.png --true-room salle3
```

Dossier de test :
```bash
python optimize_weights.py --index index.pkl --labels labels.json --test-dir ./test_images/
```

## Methodes de fusion

### Methode 1 : Lineaire (proportionnelle)

```
Score_final = alpha x S_CLIP + (1-alpha) x S_OCR
```

Alpha fixe (defaut 0.7). Simple et interpretable.

### Methode 2 : Softmax (article)

1. Scores bruts : S = [S_CLIP, S_OCR]
2. Normalisation Min-Max : S_hat = (S - min) / (max - min)
3. Poids Softmax : w = exp(S_hat / T) / sum(exp(S_hat / T))
4. Score final : S_hat^T . w

Les poids sont dynamiques : si CLIP est dominant pour une salle, il prend plus de poids automatiquement.
T = 0.7 (valeur de l'article). T bas = poids plus contrastes. T haut = poids plus equilibres.

## labels.json

```json
{
  "salle1": ["EC", "EE", "ED"],
  "salle3": ["IV", "VI.005"]
}
```

Cle = nom de la salle, valeur = textes visibles sur les portes/murs.
