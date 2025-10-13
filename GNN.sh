#!/bin/bash
# Script d'installation et d'exécution du pipeline GML Géospatial.

# Nom de l'environnement virtuel pour l'isolation
VENV_DIR=".venv"
git clone https://git.lab.sspcloud.fr/bhurpeau/gml.git
cd gml
echo "--- Démarrage de la configuration de l'environnement GML ---"

# --- 1. Installation des dépendances ---
pip install --upgrade pip
pip install uv ipykernel
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync
python -m ipykernel install --user --name=venv-gml --display-name "Python (.venv GML)"
mkdir ./data
mc cp -r  s3/bhurpeau/graphe/data/ data/