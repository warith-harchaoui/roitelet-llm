# INSTALLER — Roitelet LLM

Guide d'installation complet pour tous les modes de déploiement supportés.

---

## Prérequis

| Outil | Version minimale | Rôle |
|---|---|---|
| [Ollama](https://ollama.com) | 0.3+ | Modèle local de synthèse / juge |
| Python | 3.11+ | Exécution |
| [conda](https://docs.conda.io) **ou** venv | toute | Isolation de l'environnement |
| [Docker](https://docs.docker.com/get-docker/) | 24+ | Déploiement conteneurisé (optionnel) |

> **Bundle OSS recommandé — à télécharger avant le premier lancement :**
> ```bash
> ./scripts/pull_defaults.sh
> ```
> Roitelet fusionne K réponses en parallèle ; le script installe un
> modèle de chaque grande famille OSS (Qwen, Llama, Gemma, Phi) plus
> un modèle vision-langage. Empreinte disque totale ≈ 15 Go. Sans au
> moins le modèle de synthèse par défaut (`qwen3:8b`), l'étape
> de couronnement n'a rien à fusionner.

---

## Option A — Conda (recommandée)

### A1. Création en une commande

```bash
conda env create -f environment.yaml
conda activate roitelet-llm
```

Le fichier `environment.yaml` fixe Python 3.11 et délègue l'installation des
paquets à `requirements.txt` via pip.

### A2. Création manuelle (équivalent)

```bash
conda create -n roitelet-llm python=3.11 -y
conda activate roitelet-llm
pip install -r requirements.txt
```

---

## Option B — pip + venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Extras optionnels

`pyproject.toml` expose trois jeux de dépendances optionnels afin de
garder le démarrage rapide léger :

| Extra | Inclut | À utiliser quand |
|---|---|---|
| `dev` | pytest, pytest-asyncio, ruff | Exécution de la suite de tests ou linting |
| `eval` | deepeval | Suite d'évaluation `@pytest.mark.eval` (qualité des réponses) |
| `multimodal` | pywhispercpp, NeMo (~2 Go), soundfile, kreuzberg | Pour les pièces jointes audio / PDF via l'interface web |

Installation :

```bash
pip install -e .[dev]
pip install -e .[multimodal]      # la légende d'image est gérée côté serveur par le VLM Ollama local — pas de dépendance Python supplémentaire
```

---

## Option C — Docker

### C1. Construction et démarrage

```bash
cp .env.example .env          # puis éditez .env avec vos identifiants
docker compose up --build -d
```

Le conteneur expose :
- **API + interface web** : `http://localhost:8000` (le processus FastAPI sert l'API JSON et le client web statique sur la même origine)

> **Ollama sur la machine hôte**
> Le fichier compose configure automatiquement
> `LOCAL_LLM_BASE_URL=http://host.docker.internal:11434`
> afin que Roitelet (dans Docker) puisse accéder à Ollama tournant nativement
> sur votre machine (macOS, Windows et Linux avec Docker 20.10+).

### C2. Persistance des données

Les conversations, la télémétrie, l'état Elo et les paramètres sont écrits
dans le volume Docker nommé `roitelet_data`. Pour inspecter ou sauvegarder :

```bash
docker volume inspect roitelet_data
```

### C3. Commandes utiles

```bash
docker compose logs -f                  # logs en temps réel
docker compose ps                       # vérifier l'état de santé
docker compose down                     # arrêter
docker compose down -v                  # arrêter + supprimer le volume
docker compose pull && docker compose up -d   # mettre à jour l'image
```

---

## Configuration

### 1. Copier le modèle d'environnement

```bash
cp .env.example .env
```

### 2. Réglages minimaux conseillés

```env
# Modèles payants via OpenRouter
OPENROUTER_API_KEY=sk-or-...

# Modèle local de synthèse / juge
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen3:8b
LOCAL_VLM_MODEL=qwen2.5vl:7b

```

> **Mode local uniquement (coût zéro)**
> Vous pouvez fonctionner entièrement hors ligne, sans clé API. Définissez
> `ROITELET_CANDIDATE_POOL_SIZE=4` et ajoutez des modèles Ollama via
> la page de configuration web.

### 3. Référence complète des variables

Consultez [`.env.example`](.env.example) pour toutes les variables disponibles et leurs valeurs par défaut.

---

## Démarrage du service

### Démarrage direct (conda ou venv)

```bash
chmod +x start.sh
./start.sh
```

Cela lance un unique processus uvicorn sur `http://localhost:8000` qui sert à la fois l'API JSON et le client web statique sur `/`.

### Lancement manuel

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Vérification au premier lancement

```bash
# 1. Contrôle de santé de l'API
curl http://localhost:8000/

# 2. Lister les modèles enregistrés
curl http://localhost:8000/v1/models

# 3. Envoyer un prompt de test (Ollama doit être lancé)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Quelle est la capitale de la France ?", "top_k": 1}'
```

Réponse attendue du contrôle de santé :
```json
{"status": "ok", "service": "roitelet-llm", "base_url": "http://localhost:8000"}
```

---

## Lancer la suite de tests

```bash
# Installer les dépendances de développement
pip install -e .[dev]

# Suite par défaut (sans réseau, ~1 s)
pytest tests/ -q

# Optionnel : suite d'évaluation qualité (requiert un juge Ollama local)
pip install -e .[eval]
pytest -m eval -q
```

La suite par défaut ignore les tests marqués `@pytest.mark.eval` (lents,
dépendants du réseau). Voir `[tool.pytest.ini_options]` dans
`pyproject.toml` pour la liste des marqueurs.

---

## Mise à jour

### pip / conda

```bash
git pull
pip install -r requirements.txt   # récupérer les nouveaux paquets éventuels
./start.sh
```

### Docker

```bash
git pull
docker compose build --no-cache
docker compose up -d
```

---

## Résolution de problèmes

| Symptôme | Cause probable | Solution |
|---|---|---|
| `FileNotFoundError: Bootstrap priors not found` | Clone corrompu | Re-cloner le dépôt |
| `Connection refused` sur le port 8000 | API non démarrée | Lancer `./start.sh` |
| La synthèse retourne toujours vide | Ollama non démarré | `ollama serve` |
| `401 Unauthorized` depuis OpenRouter | Clé incorrecte | Mettre à jour `OPENROUTER_API_KEY` dans `.env` |
| Les modèles n'apparaissent pas après `ollama pull` | TTL du cache | Attendre jusqu'à 60 s ou redémarrer l'API |

---

## Arborescence du projet

```text
roitelet-llm/
├── core/               # routeur, registre, juge, pipeline, capacités
│   ├── providers/      # ollama, openai-compatible (OpenRouter, OpenAI, ...)
│   └── multimodal/     # audio (whisper.cpp + NeMo), image (VLM Ollama), pdf (kreuzberg)
├── api/                # FastAPI (natif + OpenAI-compatible + MCP + multimodal)
├── web/                # Client web statique servi sur `/` par l'API
├── cli/                # Interface en ligne de commande (REPL)
├── docs/               # Guides ciblés (ex. ADDING_PAID_LLM.md)
├── data/
│   └── bootstrap/model_priors.json   # scores a priori inspirés des benchmarks
├── tests/              # Suite pytest — core, api, pipeline, cli, scripts, eval
├── assets/             # Branding (logo)
├── start.sh            # script de lancement
├── Dockerfile          # construction multi-étapes
├── docker-compose.yml  # pile compose
├── environment.yaml    # environnement conda
├── requirements.txt    # dépendances pip
├── pyproject.toml      # build + extras optionnels (dev / eval / multimodal)
├── MECHANISM.md        # Architecture détaillée (diagrammes Mermaid)
└── .env.example        # modèle de variables d'environnement
```
