@echo off
echo ========================================
echo  Lancement de l'interface Streamlit
echo ========================================
echo.

REM Vérifier si streamlit est installé
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [ERREUR] Streamlit n'est pas installe.
    echo Installation des dependances...
    pip install -r requirements.txt
    echo.
)

REM Vérifier si le fichier .env existe
if not exist .env (
    echo [ATTENTION] Le fichier .env n'existe pas.
    echo Copiez .env.example en .env et ajoutez votre cle API Google Gemini.
    echo.
    pause
)

echo Lancement de l'application...
echo L'application s'ouvrira dans votre navigateur a l'adresse : http://localhost:8501
echo.
echo Appuyez sur Ctrl+C pour arreter l'application.
echo.

streamlit run app.py

