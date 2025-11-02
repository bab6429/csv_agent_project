#!/bin/bash

echo "========================================"
echo " Lancement de l'interface Streamlit"
echo "========================================"
echo ""

# Vérifier si streamlit est installé
if ! python -c "import streamlit" 2>/dev/null; then
    echo "[ERREUR] Streamlit n'est pas installé."
    echo "Installation des dépendances..."
    pip install -r requirements.txt
    echo ""
fi

# Vérifier si le fichier .env existe
if [ ! -f .env ]; then
    echo "[ATTENTION] Le fichier .env n'existe pas."
    echo "Copiez .env.example en .env et ajoutez votre clé API Google Gemini."
    echo ""
    read -p "Appuyez sur Entrée pour continuer..."
fi

echo "Lancement de l'application..."
echo "L'application s'ouvrira dans votre navigateur à l'adresse : http://localhost:8501"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter l'application."
echo ""

streamlit run app.py

