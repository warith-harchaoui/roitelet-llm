# Roitelet LLM

> **Un atelier local-first de routage et de fusion de LLM.** Roitelet
> route les prompts vers des modèles locaux et distants, compare leurs
> réponses, synthétise une réponse finale localement, et apprend ses
> préférences de routage au fil du temps à partir de son propre signal
> de juge.

Ce n'est **pas** un « LLM universel », ni une garantie d'obtenir la
meilleure réponse pour toute requête. C'est une plateforme
d'expérimentation pour les compromis coût / latence / vie privée /
qualité qui apparaissent dès qu'on cesse de s'engager sur un seul
modèle.

![Roitelet](assets/roitelet.jpg)

---

## Ce que Roitelet fait

Pour un prompt donné, Roitelet :

1. Note chaque modèle enregistré (local + distant optionnel) à l'aide
   d'un routeur hybride — a priori de capacité curés, Elo glissant, et
   un petit jeu d'ajustements régime-conscients (budget coût,
   prompt trivial, long contexte, …).
2. Fan-out parallèle sur les top-K candidats (K=3 par défaut).
3. Transmet les K réponses, **anonymisées et mélangées**, à un juge
   de synthèse local qui les fusionne en une réponse unique.
4. Persiste la télémétrie par tour et ajuste légèrement les scores
   Elo par capacité pour améliorer le routage du prompt suivant.

Chaque étape est inspectable. La décision du routeur, les réponses
candidates, le raisonnement du juge et l'état Elo glissant atterrissent
sous forme de JSON sur le disque ; rien n'est caché derrière un
service opaque.

### Ce pourquoi c'est utile

- **Comparer des familles de modèles** sur un même prompt sans
  jongler avec trois SDK.
- **Lancer une passe de synthèse locale** par-dessus les réponses
  candidates distantes — utile quand on veut que le mot final vienne
  d'un modèle que l'on contrôle.
- **Expérimenter avec des stratégies de routage et de fusion**
  (filtres de budget, routeur appris par factorisation matricielle,
  détecteur de capacité par embeddings) sous une même API.
- **Étudier les compromis** entre coût, latence, vie privée et
  qualité de réponse, avec la traçabilité nécessaire pour rendre ces
  études reproductibles.

### Ce que ce projet ne prétend **pas**

- Que la réponse fusionnée est toujours meilleure que le meilleur
  candidat individuel. La fusion aide ou non selon la classe du
  prompt, le modèle juge et la diversité des candidats ; c'est
  précisément ce que mesure la feuille de route d'ablation dans
  [docs/EVALUATION.md](docs/EVALUATION.md).
- Que le juge de synthèse local soit un oracle objectif. Roitelet
  apprend des préférences *conditionnées au juge* — différents juges
  produisent différentes trajectoires d'Elo. Ce biais du juge est
  une caractéristique à inspecter, pas un bug à cacher.
- Qu'il soit automatiquement « privé ». Roitelet est local-**first**,
  pas local-**only**. Les prompts peuvent toujours sortir vers des
  fournisseurs distants quand ils sont sélectionnés comme candidats.
  Voir [docs/PRIVACY.md](docs/PRIVACY.md) pour la distinction
  précise.

---

## Le roitelet

Le projet porte le nom du roitelet, un tout petit oiseau qui, dans
la fable, se cache dans le plumage de l'aigle et bat des ailes
légèrement plus haut au dernier moment. La métaphore parle de
composer de petits mouvements locaux par-dessus de grands modèles
externes — pas de prétendre être le meilleur oiseau de la forêt.

---

## Fonctionnalités

- **Routage hybride.** A priori de capacité + Elo glissant +
  ajustements régime-conscients (budget coût, prompt trivial, long
  contexte, ambigu, capacité dominante). Routeur appris optionnel par
  factorisation matricielle (`ROITELET_ROUTER=mf`).
- **Fan-out parallèle top-K.** K=3 par défaut, configurable par tour.
  Le temps mur est borné par le candidat le plus lent (voir la
  section [latence et coût](#latence-et-coût) ci-dessous).
- **Passe de synthèse locale.** Les réponses candidates sont
  anonymisées, mélangées et transmises à un modèle local Ollama qui
  les fusionne. Le juge est remplaçable.
- **Elo glissant par capacité.** À chaque tour, les gagnants du juge
  gagnent de l'Elo sur les capacités appelées par le prompt ; les
  perdants en perdent. Mises à jour bornées ; pas d'emballement.
- **Point d'extension universel.** Tout LLM payant avec une API
  `/v1/chat/completions` compatible OpenAI s'enregistre via trois
  champs de réglage. Idem pour tout GGUF local servi par
  `llama-server`.
- **Pièces jointes multimodales.** Glissez images, PDF, audio —
  extraits localement (Ollama VLM, kreuzberg, whisper.cpp + NeMo)
  avant la pipeline textuelle.
- **Génération d'images.** Routage K=1 vers le modèle image-gen
  enregistré le plus apte (pas de fusion — l'ensemblage d'images
  n'est pas défini).
- **Mode personnel.** Déposez vos propres fichiers dans un dossier ;
  petits corpus injectés en long-context (style LLM-wiki à la
  Karpathy), gros corpus basculés sur recherche par embeddings.
  Inclut une projection PCA 2-D. Voir
  [docs/PERSONAL_MODE.md](docs/PERSONAL_MODE.md).
- **Deux détecteurs de capacité.** Scan par mots-clés par défaut +
  classifieur sur embeddings locaux Ollama en option.
- **Commandes slash.** `/image`, `/speech`, `/personal`, `/local`,
  `/cheap <usd>`, `/k <n>`, `/help`. Voir
  [docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md).
- **Endpoints standardisés.** `/v1/chat/completions` +
  `/v1/images/generations` compatibles OpenAI, FastAPI natif, MCP
  JSON-RPC.
- **Télémétrie locale.** Enregistrement JSON par tour de la décision
  du routeur, de chaque réponse candidate (y compris les échecs),
  de la synthèse et des gagnants. Voir
  [docs/PRIVACY.md](docs/PRIVACY.md) pour ce qui est enregistré.
- **Gate Bearer-token optionnel.** `ROITELET_API_TOKEN` verrouille
  tous les endpoints mutants ou de listage. Désactivé par défaut
  pour préserver l'UX mono-utilisateur en local.

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
├── INSTALLER.md        # Guide d'installation (FR)
├── INSTALL.md          # Guide d'installation (EN)
├── MECHANISM.md        # Architecture détaillée (Mermaid) — contributeurs
└── .env.example
```

---

## Plan de la documentation

La documentation est organisée en trois niveaux. Choisissez celui qui
correspond à ce que vous voulez faire.

### Niveau 1 — Utilisateurs (vous voulez *lancer* Roitelet)
- **[LISEZMOI.md](LISEZMOI.md)** / **[README.md](README.md)** — ce
  qu'est Roitelet, pourquoi il existe, démarrage rapide en 5 minutes.
- **[INSTALLER.md](INSTALLER.md)** / **[INSTALL.md](INSTALL.md)** —
  guide d'installation complet (conda, venv, Docker).

### Niveau 2 — Tech (vous voulez *utiliser* les fonctionnalités)
- **[docs/ADDING_PAID_LLM.md](docs/ADDING_PAID_LLM.md)** — brancher
  n'importe quel LLM payant compatible OpenAI (ChatGPT, Mistral, …).
- **[docs/ADDING_LOCAL_LLM.md](docs/ADDING_LOCAL_LLM.md)** — apporter
  votre propre GGUF via Ollama ou `llama-server`.
- **[docs/IMAGE_GENERATION.md](docs/IMAGE_GENERATION.md)** — activer
  la génération d'images (DALL-E, Stable Diffusion local, …).
- **[docs/PERSONAL_MODE.md](docs/PERSONAL_MODE.md)** — déposer des
  fichiers, ingestion, interroger votre base personnelle. Inclut la
  projection 2-D des embeddings (style Karpathy).
- **[docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md)** — `/image`,
  `/personal`, `/local`, `/cheap`, `/k`, `/help` — surcharges par tour.

### Niveau 3 — Contributeurs (vous voulez *modifier* Roitelet)
- **[MECHANISM.md](MECHANISM.md)** — visite architecturale complète
  avec diagrammes Mermaid. Maths de routage, régimes, boucle Elo,
  les deux routeurs, les deux détecteurs de capacité, pipeline
  image-gen.

---

## Licence

Distribué sous **licence BSD 3-Clause** — voir [LICENSE](LICENSE).

## Auteur

[Warith HARCHAOUI](https://www.linkedin.com/in/warith-harchaoui/)
