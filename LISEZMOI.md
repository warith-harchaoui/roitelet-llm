# Roitelet LLM

> **Le LLM Universel :** Le meilleur grand modèle de langage pour votre prompt, quoi qu'il arrive.

Chaque semaine, un nouveau modèle toujours plus puissant est publié par un géant de l'IA. Évaluer, comparer et maintenir l'intégration avec chacun d'entre eux est épuisant.

**Roitelet** est un routeur adaptatif qui fait le tri pour vous. Au lieu de choisir manuellement un modèle, vous interrogez Roitelet. Le système détecte la nature de votre question, prédit les 3 meilleurs modèles pour cette tâche, les exécute en parallèle, et utilise un modèle open source local pour fusionner et livrer la meilleure réponse possible.

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
- ⚡ **Synthèse Locale :** Le juge final tourne en local via Ollama, garantissant la confidentialité et le contrôle de l'arbitrage.
- 🌍 **Intégrations Natives :** Support d'OpenRouter, des points de terminaison compatibles OpenAI, Anthropic, Gemini, Perplexity, etc.
- 📊 **Monitoring Coût / Énergie :** Dashboard intégré pour suivre la consommation de tokens, évaluer l'énergie (kWh) et l'empreinte carbone (gCO₂e).
- 🔄 **Apprentissage Continu :** Un système de mises à jour basées sur un score d'évaluation Elo roule en permanence pour prioriser les modèles les plus pertinents au fil du temps.
- 🔌 **API Standard :** Expose une route `/v1/chat/completions` (OpenAI-compatible), un point de terminaison natif FastAPI, ainsi qu'un serveur JSON-RPC (MCP).

---

## Interface Utilisateur & Contrôle

Roitelet est fourni avec un tableau de bord **Streamlit** offrant une vue transparente sur votre flotte d'IA :

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

# 3. Télécharger le modèle de synthèse / couronnement
ollama pull qwen2.5:14b-instruct

# 4. Lancer l'application
chmod +x start.sh
./start.sh
```

- **API :** `http://localhost:8000`
- **Interface Streamlit :** `http://localhost:8501`

---

## Arborescence du projet

```text
roitelet-llm/
├── app/
│   ├── core/           # routeur, registre, juge, pipeline, capacités
│   ├── providers/      # clients Ollama et intégrations diverses
│   ├── config.py       # paramètres (pydantic-settings)
│   ├── main.py         # application FastAPI
│   ├── schemas.py      # modèles Pydantic partagés
│   └── storage.py      # couche de persistance structurée (JSON)
├── data/
│   └── bootstrap/model_priors.json   # Base d'informations avec scores Elo
├── scripts/            # Crawler de données d'évaluation
├── tests/
│   └── test_roitelet.py              # Suite de tests Pytest automatisés
├── streamlit_app.py    # Interface Streamlit
├── start.sh            # Script de lancement
├── Dockerfile          # Fichier de build Docker multi-stade
├── docker-compose.yml  # Déploiement en conteneur
├── environment.yaml    # Dépendances Conda
├── requirements.txt    # Dépendances natives Python (pip)
├── INSTALLER.md        # Guide manuel avancé (FR)
├── INSTALL.md          # Guide manuel avancé (EN)
└── .env.example
```

---
© 2025 deraison.ai | `warithmetics@deraison.ai`
