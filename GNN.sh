#!/bin/bash
# Script d'installation et d'exécution du pipeline GML Géospatial.

# Nom de l'environnement virtuel pour l'isolation
VENV_DIR=".venv"
cd GML
echo "--- Démarrage de la configuration de l'environnement GML ---"

# --- 1. Installation des dépendances ---
pip install uv ipykernel
# Vérification de l'outil 'uv' (le plus rapide et moderne)
if command -v uv &> /dev/null
then
    echo "Utilisation de l'outil moderne 'uv' pour la gestion des dépendances..."
    # Création de l'environnement virtuel
    uv venv "$VENV_DIR"
    
    # Activation de l'environnement
    source "$VENV_DIR/bin/activate"
    
    # Installation des dépendances à partir de pyproject.toml
    uv sync
else
    echo "L'outil 'uv' n'est pas trouvé. Retour à la méthode standard 'pip'."
    
    # Création et activation VENV avec python
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Installation des dépendances à partir de pyproject.toml
    pip install --upgrade pip
    pip install .  # Installe les dépendances du dossier courant (pyproject.toml)
fi

# Vérification et installation des dépendances spécifiques si échec
if [ $? -ne 0 ]; then
    echo "Échec de l'installation avec pyproject.toml. Tentative d'installation directe..."
    pip install torch==2.5.0 torch-geometric>=2.6.1 geopandas networkx python-louvain scikit-learn pandas
fi
python -m ipykernel install --user --name=venv-gml --display-name "Python (.venv GML)"

mc cp -r  s3/bhurpeau/graphe/data/* ./data/