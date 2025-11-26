#!/bin/bash
# Script d'installation et d'exécution du pipeline GML Géospatial.

# Nom de l'environnement virtuel pour l'isolation
VENV_DIR=".venv"
echo "--- Démarrage de la configuration de l'environnement GML ---"

# --- 1. Installation des dépendances ---
pip install --upgrade pip
pip install uv ipykernel
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync
uv pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
python -m ipykernel install --user --name=venv-gml --display-name "Python (.venv GML)"
mkdir ./data
mkdir ./out
mc cp -r  s3/bhurpeau/graphe/data/ data/