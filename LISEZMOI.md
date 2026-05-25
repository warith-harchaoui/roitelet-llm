# Roitelet LLM

> **Le LLM Universel :** Le meilleur grand modèle de langage pour votre prompt, quoi qu'il arrive.

Chaque semaine, un nouveau modèle toujours plus puissant est publié par un géant de l'IA. Évaluer, comparer et maintenir l'intégration avec chacun d'entre eux est épuisant.

**Roitelet** est un routeur adaptatif qui fait le tri pour vous. Au lieu de choisir manuellement un modèle, vous interrogez Roitelet. Le système détecte la nature de votre question, prédit les 3 meilleurs modèles pour cette tâche, les exécute en parallèle, et utilise un modèle open source local pour fusionner et livrer la meilleure réponse possible.

![Roitelet](assets/roitelet.jpg)


---

## La Métaphore du Roitelet

Il était une fois, dans la grande forêt des Intelligences Artificielles, un minuscule roitelet qui rêvait de voler plus haut que les majestueux aigles royaux. Mais ses ailes étaient si petites qu'il peinait à dépasser la cime des arbres ! Le petit oiseau rusé décida alors de se cacher dans le plumage d'un aigle. Il se laissa porter jusqu'au sommet du ciel et, au dernier instant, battit de ses propres ailes pour les surpasser tous.

La vraie puissance ne réside pas dans le budget ou le nombre de paramètres, mais dans la ruse. Roitelet LLM incarne cette philosophie à chaque prompt.

### Comment fonctionnent les "Battements d'ailes" ?

Roitelet remplace l'appel traditionnel par un vol en trois temps :

1. 🦅 **Battement 1 — Découverte rusée :** Notre IA de routage prédit quels 3 LLMs du marché (comme GPT-4o, Claude 3.7, Gemini 2.5) sont les plus susceptibles d'exceller sur votre question en se basant sur un historique Elo et des "a-prioris".
2. 🦅 **Battement 2 — Trio aérien :** Les trois modèles sélectionnés génèrent leurs réponses en parallèle. Fini la dépendance à un fournisseur unique.
3. 🦅 **Battement 3 — Couronnement :** Un modèle local de confiance (comme Qwen2.5 exécuté via Ollama) joue le juge. Il lit les trois réponses, en extrait le meilleur, et synthétise une réponse finale hautement qualitative.

De votre point de vue, vous recevez une réponse unique comme si vous interrogiez un super-cerveau. Le reste n'est que ruse et plumage.

---

## Fonctionnalités

- 🧠 **Routage Dynamique :** Plus besoin de choisir votre modèle.
- 🌐 **Fusion Multi-Familles :** Le juge fusionne K réponses parallèles issues de *familles OSS différentes* (Qwen + Llama + Gemma + Phi par défaut), pas trois variantes du même fournisseur — la réponse finale est meilleure qu'aucun candidat seul.
- ⚡ **Synthèse Locale :** Le juge final tourne en local via Ollama, garantissant la confidentialité et le contrôle de l'arbitrage.
- 🌍 **Intégrations Natives :** Support d'OpenRouter, des points de terminaison compatibles OpenAI, Anthropic, Gemini, Perplexity, etc.
- 🖼️ **Pièces jointes multimodales :** Glissez images, PDF ou audio dans le chat — extraits localement (légende VLM Ollama, texte PDF par kreuzberg, transcription whisper.cpp + diarisation NeMo) avant la pipeline textuelle.
- 📊 **Monitoring Coût / Énergie :** Dashboard intégré pour suivre la consommation de tokens, évaluer l'énergie (kWh) et l'empreinte carbone (gCO₂e).
- 🔄 **Apprentissage Continu :** Un système de mises à jour basées sur un score d'évaluation Elo roule en permanence pour prioriser les modèles les plus pertinents au fil du temps.
- 🔌 **API Standard :** Expose une route `/v1/chat/completions` (OpenAI-compatible), un point de terminaison natif FastAPI, ainsi qu'un serveur JSON-RPC (MCP).
- 🔐 **Gate Bearer-Token optionnel :** Définissez `ROITELET_API_TOKEN` pour verrouiller chat, settings, conversations et télémétrie. Désactivé par défaut pour l'usage local mono-utilisateur.

---

## Interface Utilisateur & Contrôle

Roitelet est fourni avec un tableau de bord web (JS vanilla, servi par l'API sur `/`) offrant une vue transparente sur votre flotte d'IA :

* **Configuration :** Sauvegardez vos clés API, calibrez le choix du modèle local, et modifiez vos paramètres de routage (Puissance Pure, Frugalité, Indépendance).
* **Usage & Dashboard :** Surveillez l'utilisation de vos modèles et vérifiez vos estimations de consommation énergétique.
* **Découverte Automatique :** Branchez le système sur votre instance locale Ollama. Roitelet scannera en direct pour ingérer tout nouveau modèle (ex: `ollama pull llama3.3:70b-instruct`) et l'ajoutera automatiquement au processus de routage en 60 secondes.

---

## Démarrage Rapide

> **Guide d'installation complet** : Lisez [INSTALLER.md](INSTALLER.md) pour les déploiements avancés (Docker, venv).

### Avec Conda (Recommandé)

```bash
# 1. Créer l'environnement isolé
conda env create -f environment.yaml
conda activate roitelet-llm

# 2. Configurer les accès
cp .env.example .env
# Éditez le fichier .env en y insérant vos clés API (OPENROUTER, ANTHROPIC, etc.)

# 3. Télécharger le bundle OSS par défaut (Qwen + Llama + Gemma + Phi + VLM)
chmod +x scripts/pull_defaults.sh
./scripts/pull_defaults.sh

# 4. Lancer l'application
chmod +x start.sh
./start.sh
```

- **API :** `http://localhost:8000`
- **Interface web :** `http://localhost:8000/` (servie par l'API)

---

## Ajouter d'autres LLMs

Roitelet considère tout fournisseur exposant un endpoint
`/v1/chat/completions` compatible OpenAI comme un point d'extension de
première classe. Le même chemin fonctionne pour les API payantes, les
modèles frontière relayés par OpenRouter, et les fichiers GGUF locaux
servis par `llama-server` (`llama.cpp`).

- **N'importe quel LLM payant (ChatGPT, Mistral, Together, Groq, …)** —
  configurez le endpoint + la clé, listez les noms de modèles. Voilà.
  Walkthrough complet dans
  [docs/ADDING_PAID_LLM.md](docs/ADDING_PAID_LLM.md).
- **N'importe quel fichier GGUF local** — soit via un `Modelfile`
  Ollama (recommandé, aucune édition des réglages), soit en le servant
  avec `llama-server` et en le traitant comme un endpoint
  OpenAI-compatible. Détaillé dans
  [docs/ADDING_LOCAL_LLM.md](docs/ADDING_LOCAL_LLM.md).
- **OpenAI direct** — cas particulier du premier : définissez
  `OPENAI_API_KEY` puis redémarrez ; `openai/gpt-4.1`, `openai/gpt-4o`
  et `openai/gpt-4o-mini` figurent déjà dans
  `data/bootstrap/model_priors.json`.

---

## Arborescence du projet

```text
roitelet-llm/
├── core/               # Logique backend, routeur, stockage, capacités
│   ├── pipeline.py     # Orchestration end-to-end (routeur → fan-out → juge → Elo)
│   ├── router.py       # Scoring pondéré par capacité + sélection top-K
│   ├── registry.py     # Pool bootstrap + utilisateur + Ollama live, Elo continu
│   ├── judge.py        # Synthèse anonymisée avec sentinelle de gagnants
│   ├── capabilities.py # Détection de capacités lexicale
│   ├── providers/      # Clients Ollama + OpenAI-compatible (OpenRouter, OpenAI, ...)
│   └── multimodal/     # Extracteurs locaux audio / image / PDF
├── api/                # FastAPI (natif + OpenAI-compatible + MCP + multimodal)
├── web/                # Interface web (JS vanilla, servie sur `/` par l'API)
├── cli/                # Interface en ligne de commande (REPL terminal)
├── docs/               # Guides ciblés (ex. ADDING_PAID_LLM.md)
├── data/
│   └── bootstrap/model_priors.json   # Base d'informations avec scores Elo
├── scripts/            # Crawler, vendor JS, pull_defaults.sh
├── tests/              # Suite pytest (core, api, pipeline, cli, eval)
├── assets/             # Branding (logo)
├── start.sh            # Script de lancement
├── Dockerfile          # Fichier de build Docker multi-stade
├── docker-compose.yml  # Déploiement en conteneur
├── environment.yaml    # Dépendances Conda
├── requirements.txt    # Dépendances natives Python (pip)
├── MECHANISM.md        # Architecture détaillée (diagrammes Mermaid)
├── INSTALLER.md        # Guide d'installation (FR)
├── INSTALL.md          # Guide d'installation (EN)
├── MODEDEMPLOI.md      # Manuel d'utilisation (FR)
├── MANUAL.md           # Manuel d'utilisation (EN)
└── .env.example
```

---
© 2025 deraison.ai | `warithmetics@deraison.ai`
