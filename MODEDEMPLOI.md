# MODE D’EMPLOI — Installation et réglages

> **Guide d'installation complet** : [INSTALLER.md](INSTALLER.md)

## 1. Installation avec conda

```bash
# Option recommandée — en une commande
conda env create -f environment.yaml
conda activate roitelet-llm

# Alternative
conda create -n roitelet-llm python=3.11 -y
conda activate roitelet-llm
pip install -r requirements.txt
```

## 2. Création du fichier `.env`

```bash
cp .env.example .env
```

Puis éditez `.env`.

### Configuration minimale conseillée

```env
OPENROUTER_API_KEY=...
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen3:8b
LOCAL_VLM_MODEL=qwen2.5vl:7b
```

## 3. Lancer le projet

```bash
./start.sh
```

Cela lance un unique processus FastAPI sur le port `8000` qui sert l'API JSON et le tableau de bord web sur `/`.

## 4. Déploiement Docker

```bash
docker compose up --build
```

## 5. Interface web

### Page de configuration
Vous pouvez :
- renseigner les clés API,
- choisir l’URL d’Ollama,
- choisir le modèle local de synthèse,
- autoriser les VLM,
- régler les curseurs :
  - **Raw Power**,
  - **Frugality**,
  - **Independence**.

### Page de monitoring
Vous voyez :
- les usages par modèle,
- le champ dominant du prompt,
- la latence,
- le coût estimé,
- l’énergie estimée,
- le CO₂ estimé.

### Pièces jointes (multimodal)
Le champ de saisie accepte des pièces jointes. Chaque fichier est converti en
texte **localement** avant que la pipeline standard ne s'exécute — le routeur,
le fan-out des candidats et le juge restent purement textuels :

| Format | Pipeline | Notes |
|---|---|---|
| Image (`.png`, `.jpg`, `.webp`, ...) | Légende via VLM Ollama local (`qwen2.5vl:7b` par défaut) | Contrôlée par l'interrupteur **Allow VLMs** de la page Config |
| PDF (`.pdf`) | `kreuzberg` (texte natif pdfium, OCR Tesseract en secours) | Nécessite `pip install -e .[multimodal]` |
| Audio (`.wav`, `.mp3`, `.m4a`, ...) | Transcription `pywhispercpp` + diarisation NeMo Sortformer | Nécessite `pip install -e .[multimodal]` |

Les formats inconnus sont ignorés et la mention `[Note] Skipped: ...` est
ajoutée au prompt afin de rendre l'ignorance visible.

## 6. Utilisation en ligne de commande (CLI)

Mode REPL interactif (inspiré par Gemini CLI) :
```bash
python -m cli chat
```

Mode requête unique :
```bash
python -m cli ask "Quelle est la capitale de la France ?"
```

## 7. API compatible OpenAI

Point d’entrée :

```text
POST /v1/chat/completions
```

Nom du modèle :

```text
roitelet-llm
```

## 7. Accès MCP

Point d’entrée :

```text
POST /mcp
```

Méthodes disponibles :
- `initialize`
- `tools/list`
- `tools/call`

Outil principal :
- `roitelet.chat`

## 8. Données locales

Le projet écrit des fichiers JSON dans `data/` :
- `conversations/`
- `telemetry/`
- `runtime/settings.json`
- `runtime/elo_state.json`

## 9. Apprentissage continu

Pour l’instant, l’apprentissage continu est volontairement simple :
**mise à jour partielle des scores Elo** en fonction des gagnants observés et des capacités détectées dans le prompt.

C’est la bonne première étape avant un routeur plus avancé de type classifieur contextuel ou bandit.

## 10. Verrouiller l'API (déploiements LAN / multi-utilisateurs)

Le mode local mono-utilisateur par défaut laisse tous les endpoints ouverts.
Pour un déploiement partagé, définissez `ROITELET_API_TOKEN` dans `.env` :

```env
ROITELET_API_TOKEN=changez-moi
```

Une fois défini, tous les endpoints de chat, settings, conversations et
télémétrie exigent `Authorization: Bearer <token>`. `/healthz`, `/v1/models`
et les fichiers statiques restent publics pour les sondes de santé.

> **À noter** — l'interface web fournie n'envoie pas encore l'en-tête bearer.
> Avec un token configuré, préférez le CLI, l'endpoint OpenAI-compatible ou
> un reverse proxy qui injecte l'en-tête, en attendant que l'UI gère le token.
