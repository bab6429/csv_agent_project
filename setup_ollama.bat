@echo off
echo ========================================
echo   Configuration Ollama pour Agent CSV
echo ========================================
echo.

REM Vérifier si Ollama est installé
echo [1/4] Verification de l'installation d'Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ATTENTION] La commande 'ollama' n'est pas trouvee dans le PATH.
    echo.
    echo Cela peut signifier :
    echo   1. Ollama n'est pas installe
    echo   2. Ollama est installe mais pas dans le PATH
    echo.
    echo Si Ollama est installe ailleurs, vous pouvez :
    echo   - Ajouter Ollama au PATH systeme
    echo   - OU utiliser directement l'URL de l'API dans le fichier .env
    echo.
    echo Voulez-vous continuer quand meme ? (O/N)
    set /p CONTINUE_CHOICE="[O/N]: "
    if /i not "%CONTINUE_CHOICE%"=="O" (
        echo.
        echo Installation annulee.
        echo.
        echo Pour installer Ollama : https://ollama.com/download/windows
        pause
        exit /b 1
    )
    echo.
    echo [INFO] Continuation sans commande ollama...
    echo Vous devrez configurer manuellement l'URL dans le fichier .env
    echo.
    set OLLAMA_CMD_AVAILABLE=0
) else (
    echo [OK] Ollama est installe et accessible
    set OLLAMA_CMD_AVAILABLE=1
)
echo.

REM Afficher les modèles disponibles
echo [2/4] Modeles disponibles :
if %OLLAMA_CMD_AVAILABLE%==1 (
    ollama list
) else (
    echo [INFO] Impossible d'afficher les modeles (commande ollama non disponible)
    echo Vous pouvez verifier manuellement avec : ollama list
)
echo.

REM Recommander un modèle
echo [3/4] Recommendation de modele pour PC peu puissant :
echo.
echo   - phi3 (2.3 GB) - RECOMMANDE - Tres rapide et efficace
echo   - llama3.2 (2.0 GB) - Alternative excellente
echo   - gemma2:2b (1.4 GB) - Tres leger mais qualite reduite
echo.

REM Demander quel modèle télécharger
set /p MODEL_CHOICE="Quel modele voulez-vous telecharger ? (phi3/llama3.2/gemma2:2b) [phi3]: "
if "%MODEL_CHOICE%"=="" set MODEL_CHOICE=phi3

echo.
if %OLLAMA_CMD_AVAILABLE%==1 (
    echo [4/4] Telechargement du modele %MODEL_CHOICE%...
    echo Cela peut prendre quelques minutes selon votre connexion internet...
    echo.
    
    ollama pull %MODEL_CHOICE%
    
    if %errorlevel% neq 0 (
        echo.
        echo [ERREUR] Le telechargement a echoue.
        echo Verifiez votre connexion internet et reessayez.
        pause
        exit /b 1
    )
    
    echo.
    echo [OK] Modele %MODEL_CHOICE% telecharge avec succes !
    echo.
) else (
    echo [4/4] Telechargement du modele...
    echo.
    echo [INFO] La commande ollama n'est pas disponible.
    echo Vous devez telecharger le modele manuellement :
    echo   ollama pull %MODEL_CHOICE%
    echo.
    echo Ou utilisez l'interface Ollama si elle est installee.
    echo.
)

REM Demander l'URL d'Ollama si nécessaire
echo Configuration de l'URL d'Ollama...
echo.
echo Par defaut, Ollama tourne sur : http://localhost:11434
echo Si Ollama tourne sur un autre serveur ou port, indiquez l'URL.
echo.
set /p OLLAMA_URL="URL d'Ollama [http://localhost:11434]: "
if "%OLLAMA_URL%"=="" set OLLAMA_URL=http://localhost:11434

REM Créer ou mettre à jour le fichier .env
echo.
echo Creation du fichier .env...
(
echo # Configuration Ollama
echo OLLAMA_MODEL_NAME=%MODEL_CHOICE%
echo OLLAMA_BASE_URL=%OLLAMA_URL%
echo.
echo # Clé API Google Gemini (optionnelle, fallback si Ollama n'est pas disponible)
echo # GOOGLE_API_KEY=your_key_here
) > .env

echo [OK] Fichier .env cree avec :
echo   OLLAMA_MODEL_NAME=%MODEL_CHOICE%
echo   OLLAMA_BASE_URL=%OLLAMA_URL%
echo.

REM Test rapide
if %OLLAMA_CMD_AVAILABLE%==1 (
    echo Voulez-vous tester le modele maintenant ? (O/N)
    set /p TEST_CHOICE="[O/N]: "
    if /i "%TEST_CHOICE%"=="O" (
        echo.
        echo Test du modele %MODEL_CHOICE%...
        echo Tapez votre question, puis appuyez sur Entree.
        echo Tapez /bye pour quitter.
        echo.
        ollama run %MODEL_CHOICE%
    )
) else (
    echo.
    echo [INFO] Test du modele non disponible (commande ollama non accessible)
    echo Vous pouvez tester avec : python test_ollama.py
    echo.
)

echo.
echo ========================================
echo   Configuration terminee !
echo ========================================
echo.
echo Vous pouvez maintenant lancer l'application avec :
echo   streamlit run app.py
echo.
echo Ou en ligne de commande :
echo   python csv_agent.py votre_fichier.csv
echo.
pause

