# 🔧 Configurer Ollama installé ailleurs

Si Ollama est installé dans un autre emplacement que votre projet, voici comment le configurer.

## 📍 Deux choses à configurer

### 1. La commande `ollama` (pour les scripts)
### 2. L'URL de l'API Ollama (pour l'application Python)

---

## 🔍 Étape 1 : Trouver où Ollama est installé

### Méthode 1 : Chercher dans le menu Démarrer
1. Cliquez sur Démarrer
2. Cherchez "Ollama"
3. Clic droit → "Ouvrir l'emplacement du fichier"
4. Notez le chemin (ex: `C:\Users\VotreNom\AppData\Local\Programs\Ollama`)

### Méthode 2 : Chercher manuellement
Ollama est généralement installé dans :
- `C:\Users\VotreNom\AppData\Local\Programs\Ollama`
- `C:\Program Files\Ollama`
- Ou un autre emplacement personnalisé

### Méthode 3 : Vérifier si Ollama tourne
Ouvrez un navigateur et allez sur : http://localhost:11434/api/tags

Si vous voyez une réponse JSON, Ollama tourne et l'API est accessible ! ✅

---

## ⚙️ Étape 2 : Configurer l'URL de l'API (IMPORTANT)

C'est la partie la plus importante ! Même si la commande `ollama` n'est pas dans le PATH, l'application Python peut utiliser Ollama via son API HTTP.

### Option A : Ollama tourne sur localhost (par défaut)

Créez un fichier `.env` dans votre projet avec :

```env
OLLAMA_MODEL_NAME=phi3
OLLAMA_BASE_URL=http://localhost:11434
```

### Option B : Ollama tourne sur un autre port

Si Ollama tourne sur un autre port (par exemple 11435), modifiez :

```env
OLLAMA_MODEL_NAME=phi3
OLLAMA_BASE_URL=http://localhost:11435
```

### Option C : Ollama tourne sur un autre serveur

Si Ollama tourne sur une autre machine (ex: 192.168.1.100), modifiez :

```env
OLLAMA_MODEL_NAME=phi3
OLLAMA_BASE_URL=http://192.168.1.100:11434
```

---

## 🛠️ Étape 3 : Ajouter Ollama au PATH (optionnel)

Si vous voulez utiliser la commande `ollama` dans les scripts, ajoutez-le au PATH :

### Méthode Windows (via l'interface)

1. **Trouvez le chemin d'installation d'Ollama** (voir Étape 1)
2. **Ouvrez les Variables d'environnement** :
   - Appuyez sur `Windows + R`
   - Tapez `sysdm.cpl` et Entrée
   - Onglet "Avancé" → "Variables d'environnement"
3. **Modifiez la variable PATH** :
   - Dans "Variables système", trouvez "Path"
   - Cliquez sur "Modifier"
   - Cliquez sur "Nouveau"
   - Ajoutez le chemin vers Ollama (ex: `C:\Users\VotreNom\AppData\Local\Programs\Ollama`)
   - Cliquez sur "OK" partout
4. **Redémarrez votre terminal** pour que les changements prennent effet

### Méthode PowerShell (temporaire, pour la session actuelle)

```powershell
$env:Path += ";C:\Users\VotreNom\AppData\Local\Programs\Ollama"
```

(Remplacez par votre chemin réel)

---

## ✅ Étape 4 : Vérifier la configuration

### Test 1 : Vérifier que l'API Ollama est accessible

Ouvrez PowerShell et tapez :

```powershell
python test_ollama.py
```

Ce script va :
- ✅ Vérifier que l'API Ollama est accessible
- ✅ Vérifier que le modèle est disponible
- ✅ Tester une génération

### Test 2 : Test manuel de l'API

```powershell
# Test simple avec PowerShell
Invoke-WebRequest -Uri "http://localhost:11434/api/tags" | Select-Object -ExpandProperty Content
```

Vous devriez voir une liste de modèles en JSON.

---

## 🎯 Configuration rapide (sans modifier le PATH)

Si vous ne voulez pas modifier le PATH, voici la solution la plus simple :

### 1. Créez le fichier `.env` manuellement

Dans votre projet (`C:\Users\halca\csv_agent_project`), créez un fichier `.env` avec :

```env
OLLAMA_MODEL_NAME=phi3
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. Téléchargez le modèle manuellement

Si vous avez accès à Ollama (via l'interface ou un autre terminal), téléchargez le modèle :

```bash
# Depuis n'importe quel terminal où ollama fonctionne
ollama pull phi3
```

### 3. Testez

```bash
python test_ollama.py
```

---

## 🐛 Dépannage

### Erreur : "Connection refused" ou "Cannot connect"

**Problème** : Ollama n'est pas démarré ou l'URL est incorrecte.

**Solution** :
1. Vérifiez que Ollama tourne (ouvrez l'application Ollama)
2. Vérifiez l'URL dans `.env` : `OLLAMA_BASE_URL=http://localhost:11434`
3. Testez dans le navigateur : http://localhost:11434/api/tags

### Erreur : "Model not found"

**Problème** : Le modèle n'est pas téléchargé.

**Solution** :
1. Téléchargez le modèle : `ollama pull phi3`
2. Vérifiez les modèles disponibles : `ollama list`
3. Mettez à jour `OLLAMA_MODEL_NAME` dans `.env`

### La commande `ollama` ne fonctionne pas

**Problème** : Ollama n'est pas dans le PATH.

**Solution** :
- **Option 1** : Utilisez l'URL de l'API directement (voir ci-dessus)
- **Option 2** : Ajoutez Ollama au PATH (voir Étape 3)
- **Option 3** : Utilisez le chemin complet : `C:\chemin\vers\ollama.exe pull phi3`

---

## 📝 Résumé

**Pour utiliser Ollama avec votre projet, vous avez besoin de :**

1. ✅ **Ollama installé et démarré** (peu importe où)
2. ✅ **Fichier `.env` avec l'URL correcte** :
   ```env
   OLLAMA_MODEL_NAME=phi3
   OLLAMA_BASE_URL=http://localhost:11434
   ```
3. ✅ **Modèle téléchargé** (via `ollama pull phi3` ou l'interface)

**La commande `ollama` dans le PATH est optionnelle** - l'application Python utilise l'API HTTP, pas la commande en ligne !

---

## 🚀 Test final

Une fois configuré, testez avec :

```bash
python test_ollama.py
```

Si tous les tests passent, vous pouvez lancer l'application :

```bash
streamlit run app.py
```

