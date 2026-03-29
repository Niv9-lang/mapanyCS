#!/bin/bash
# Script de lancement de l'application de navigation 3D

set -e

echo "=================================="
echo "🚀 Application de Navigation 3D"
echo "=================================="

# Obtenir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Vérifier les arguments
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage: $0 <fichier.glb> [options]"
    echo ""
    echo "Options:"
    echo "  --host HOST     Adresse de l'hôte (par défaut: 127.0.0.1)"
    echo "  --port PORT     Port (par défaut: 5000)"
    echo "  --debug         Mode debug"
    echo ""
    echo "Exemple:"
    echo "  $0 resultat.glb"
    echo "  $0 resultat.glb --port 8000 --debug"
    echo ""
    exit 1
fi

MODEL_PATH="$1"

# Convertir le chemin en absolu s'il est relatif
if [[ ! "$MODEL_PATH" = /* ]]; then
    MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")"; pwd)/$(basename "$MODEL_PATH")"
fi

# Vérifier que le fichier existe
if [ ! -f "$MODEL_PATH" ]; then
    echo "✗ Erreur: Fichier introuvable: $MODEL_PATH"
    exit 1
fi

echo "✓ Fichier trouvé: $MODEL_PATH"

# Construire la commande avec chemin absolu au script
CMD="python $SCRIPT_DIR/navigation.py --model $MODEL_PATH"

# Ajouter les options supplémentaires
for arg in "${@:2}"; do
    CMD="$CMD $arg"
done

echo ""
echo "▶️  Lancement avec: $CMD"
echo ""

# Lancer l'application depuis le répertoire du script
cd "$SCRIPT_DIR"
eval "$CMD"
