# Roitelet LLM

> **Un atelier local-first de routage et de fusion de LLM.** Roitelet
> route les prompts vers des modèles locaux et distants, compare leurs
> réponses, synthétise une réponse finale localement, et apprend ses
> préférences de routage au fil du temps à partir de son propre signal
> de juge.

![Roitelet](assets/roitelet.jpg)

---

## Le roitelet

Il était une fois, dans la forêt, les oiseaux qui décidèrent que celui
qui volerait le plus haut serait couronné roi. L'aigle s'éleva sans
effort au-dessus de tous les autres. Mais un minuscule roitelet,
caché dans ses plumes, se laissa porter tout en haut — et au sommet,
d'un petit battement d'ailes supplémentaire, lui rafla la couronne.

Ce n'est pas que le roitelet soit l'oiseau le plus fort — il ne
l'est pas. Ce qui compte, c'est ce que de petits mouvements locaux,
bien placés, peuvent ajouter par-dessus de bien plus grandes forces
externes. Roitelet LLM est bâti sur la même idée : une petite
pipeline locale qui se pose sur les grands modèles de langage — les
compose, compare leurs réponses, et passe une couche de synthèse
locale par-dessus.

### Comment cela se traduit dans la pipeline

Pour un prompt donné, Roitelet :

1. **Choisit la formation de vol.** Un routeur hybride note chaque
   modèle enregistré (local + distant optionnel) selon des a priori
   de capacité curés, l'Elo glissant et un petit jeu d'ajustements
   régime-conscients (budget coût, prompt trivial, long contexte,
   …), puis garde les top-K (K=2 par défaut — point d'équilibre
   empirique du [docs/EVALUATION.md §4.3](docs/EVALUATION.md) ;
   surchargé par tour).
2. **Les laisse voler en parallèle.** Les K candidats répondent en
   concurrence via `asyncio.gather` ; un fournisseur lent ne bloque
   pas les autres.
3. **Ajoute le battement d'aile du roitelet.** Un juge de synthèse
   local lit les K réponses — anonymisées et mélangées, pour ne pas
   pouvoir reconnaître l'identité des modèles — et les fusionne en
   une réponse unique.
4. **Se souvient de ce qui a marché.** La télémétrie par tour
   atterrit en JSON sur le disque, et l'Elo glissant par capacité
   oriente légèrement la décision de routage suivante.

Chaque étape est inspectable. La décision du routeur, les réponses
candidates, le raisonnement du juge et l'état Elo sont tous des
fichiers JSON ; rien n'est caché derrière un service opaque.

---

## À quoi cela sert

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

Quelques mises en garde utiles à connaître d'emblée :

- La réponse fusionnée n'est pas garantie de battre le meilleur
  candidat individuel sur toute classe de prompt — c'est précisément
  ce que mesure la feuille de route d'ablation dans
  [docs/EVALUATION.md](docs/EVALUATION.md).
- Le juge de synthèse n'est pas un oracle objectif. Roitelet apprend
  des préférences *conditionnées au juge* ; différents juges
  produisent différentes trajectoires d'Elo. Ce biais est une
  propriété à inspecter, pas à cacher.
- Roitelet est local-**first**, pas local-**only**. Les prompts
  partent vers des fournisseurs distants quand des candidats distants
  sont sélectionnés. Voir [docs/PRIVACY.md](docs/PRIVACY.md) pour la
  distinction précise et le mode local-only.

---

## Fonctionnalités

- **Routage hybride.** A priori de capacité + Elo glissant +
  ajustements régime-conscients (budget coût, prompt trivial, long
  contexte, ambigu, capacité dominante). Routeur appris optionnel par
  factorisation matricielle (`ROITELET_ROUTER=mf`).
- **Fan-out parallèle top-K.** K=2 par défaut (le point d'équilibre
  du §4.3), configurable par tour via `ROITELET_DEFAULT_TOP_K` ou
  le champ `top_k` de la requête.
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
- **Commandes slash** — sélection de route uniquement : `/image`,
  `/speech`, `/personal`, `/help`. Les préférences par tour
  (top-K, mode local, pseudonymisation, budget) vivent sur des
  contrôles visibles : icône curseurs dans le composeur web,
  arguments `--` du CLI, booléens `preferences` dans l'API. Voir
  [docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md).
- **Pseudonymisation** — bascule opt-in qui fait réécrire localement
  les informations personnelles (noms, lieux, organisations, contacts,
  identifiants, IP) en substituts plausibles de même origine avant
  appels distants, puis restaurés dans la réponse. Fail-closed ;
  audit attaché à chaque tour. Voir [PSEUDO.md](docs/PSEUDO.md).
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

## Pourquoi la fusion peut aider — et où se loge le biais du juge

Toute l'idée du juge de synthèse repose sur une asymétrie unique :
**évaluer et synthétiser est plus facile que créer à partir de zéro.**

Un juge qui a sous les yeux K réponses candidates déjà rédigées n'a
pas besoin de connaître la réponse ; il doit comparer des
brouillons, repérer les recouvrements, écarter les contradictions,
préserver les détails utiles, et produire une réponse fusionnée
unique. C'est une tâche fondamentalement plus petite que produire
la première réponse sans aucun échafaudage. Un modèle local
relativement modeste peut s'en sortir, pour la même raison qu'un
correcteur peut noter une pile de copies sans être capable d'écrire
la meilleure lui-même.

C'est cette asymétrie qui rend plausible le pipeline « petit modèle
local par-dessus de gros candidats distants » de Roitelet. Le travail
du juge est la curation, pas l'invention.

### Est-ce que ça aide vraiment ? (K-sweep du 2026-05-26)

Campagne complète sur les 25 prompts du dataset mixte,
**local-only** (3 petits candidats OSS — `llama3.2:3b`,
`qwen2.5:3b`, `gemma3:4b` ; juge de synthèse `qwen3:8b`), notée par
DeepEval `GEval(correctness, threshold=0.6)`. Vrai K=3 (chaque tour
K=3 fan-out aux trois candidats, Gemma incluse ; voir
[l'enquête `fan_out=2`](docs/EVALUATION.md) pour comprendre
pourquoi la première tentative ne l'avait pas) :

| K | exactitude moyenne | pass (≥0,6) | latence totale | part du juge |
|---|---|---|---|---|
| 1 | 0,87 | 23 / 25 | 32,1 s | 70 % |
| 2 | **0,95** | **25 / 25** | 55,9 s | 74 % |
| 3 | 0,96 | **25 / 25** | 112,1 s | 73 % |

**Verdict** : K=1 → K=2 = **+8 points** d'exactitude moyenne
(0,87 → 0,95) et **+2 prompts** au-dessus du seuil (tous les
prompts passent à K=2), pour **+24 s** de temps mur. K=2 → K=3
atteint le plafond de qualité sur ce dataset (+1 point, 25/25
encore) mais **double le temps mur** à 112 s. **K=2 est le point
d'équilibre** sur ce juge / ce pool ; K=3 ne vaut pas son coût.
Le multilingue est la catégorie où la fusion aide le plus
(0,33 → 0,93 → 1,00) ; le long-context régresse à K=3 (le juge
sur-curera et perd des exemples concrets).

Tous les chiffres, le détail par catégorie, les deux erreurs de
grader DeepEval à K=1 (assumées honnêtement), les mises en garde,
et le suivi K=2 → optimiser-le-juge vivent dans
[docs/EVALUATION.md §4.3](docs/EVALUATION.md). Les rapports JSON
bruts (`ksweep-20260526T*Z.json`, deux fichiers — la première
tentative qui a révélé l'interaction avec le filtre VLM, et le
re-run qui l'a corrigée) sont conservés dans le répertoire
`eval_runs/` ignoré.

### Et le juge compte — beaucoup (judge-swap à K=2, 2026-05-26)

Dataset, routeur, candidats et K figés sur le point d'équilibre §4.3 ;
seul le juge de synthèse tourne sur trois tailles (`qwen3:8b`,
`gemma3:4b`, `llama3.2:3b`). Même grader `qwen3:8b` pour les trois :

| Juge | exactitude moyenne | pass (≥0,6) | latence juge moyenne |
|---|---|---|---|
| **qwen3:8b** (8B)   | **0,93** | 24 / 25 | 38,9 s |
| **gemma3:4b** (4B)  | 0,88     | 23 / 25 | 18,4 s |
| **llama3.2:3b** (3B) | 0,72    | 19 / 25 | 20,3 s |

**Verdict** : le juge 8B bat le juge 3B de **+22 points** d'exactitude
moyenne sur les mêmes prompts avec les mêmes candidats. La majorité
du temps mur passé sur Roitelet *est* le juge — le réduire restitue
de la qualité, pas seulement de la vitesse. Le juge 4B est le point
de Pareto pour les régimes contraints en latence (−5 points pour
deux fois moins de temps juge). Les 25/25 prompts montrent un
désaccord sur l'ensemble des gagnants entre juges ; 8/25 montrent
un PASS/FAIL franchement opposé — preuve forte que **Roitelet apprend
des préférences conditionnées au juge, pas universelles**, exactement
ce que le mécanisme §1.1 annonce.

Par catégorie, les points faibles des petits juges sont **l'écriture**
(0,40 sous llama3.2:3b contre 0,95–1,00 sous les juges plus gros) et
**le multilingue** (0,47 sous gemma3:4b contre 0,97 sous qwen3:8b).
La crainte naïve « les juges préfèrent leur propre famille » n'apparaît
ici que faiblement ; le motif plus fort est un **biais anti-candidat
laconique sur les petits juges** (gemma3:4b ne choisit le candidat
`llama3.2:3b`, plus terse, que 1/25 fois). Détail par prompt et
mises en garde : [docs/EVALUATION.md §4.4](docs/EVALUATION.md). JSON
brut : `judgeswap-20260526T123130Z.json`.

Cela dit, **ce n'est pas magique** :

- Roitelet apprend des préférences *conditionnées au juge*. Si Qwen
  est le juge local, la boucle Elo glissante internalise discrètement
  ce que Qwen tend à préférer. Utile pour router *sous ce juge* ;
  pas un signal universel de qualité.
- Un juge mal calibré fusionne avec assurance dans la mauvaise
  direction. Le parse fail-closed du marqueur de gagnants
  (`core/judge.py`) borne la dégradation de l'état Elo, mais le
  *contenu* d'une mauvaise fusion reste mauvais.
- Que la fusion de trois candidats OSS batte le meilleur candidat
  payant dépend de la classe du prompt, de la diversité des
  candidats et du juge. C'est empirique, pas théorique.

Les études d'ablation sont donc une préoccupation de premier ordre,
pas une note de bas de page. Voir
[docs/EVALUATION.md](docs/EVALUATION.md).

---

## Interface Utilisateur & Contrôle

Roitelet est fourni avec un tableau de bord web (JS vanilla, servi par l'API sur `/`) offrant une vue transparente sur votre flotte d'IA :

* **Configuration :** Sauvegardez vos clés API, calibrez le choix du modèle local, et modifiez vos paramètres de routage (Puissance Pure, Écofrugalité, Indépendance).
* **Usage & Dashboard :** Surveillez l'utilisation de vos modèles et vérifiez vos estimations de consommation énergétique.
* **Découverte Automatique :** Branchez le système sur votre instance locale Ollama. Roitelet scannera en direct pour ingérer tout nouveau modèle (ex: `ollama pull llama3.3:70b-instruct`) et l'ajoutera automatiquement au processus de routage en 60 secondes.

![Interface web Roitelet — un prompt utilisateur, trois candidats OSS locaux exécutés en parallèle, le juge local fusionne leurs réponses et indique celles qu'il a effectivement utilisées.](assets/screenshot.png)

---

## Note sur la sécurité

Roitelet est **sécurisé par défaut** côté réseau : `start.sh` (et la
valeur par défaut `Settings.app_host` en bare-metal) écoute sur
`127.0.0.1`, et `ROITELET_API_TOKEN` est vide. Localhost-only sans
authentification convient à un usage laptop mono-utilisateur.

L'image Docker est la seule exception — uvicorn dans le conteneur
écoute sur `0.0.0.0` car le port-forwarding du conteneur l'exige.
L'exposition réseau réelle est alors gouvernée par votre mapping
de ports dans `docker-compose.yml` et le pare-feu hôte, pas par
l'adresse de bind interne.

Si vous voulez exposer l'API sur un LAN, une IP publique, une VM
avec port-forward, ngrok, Tailscale, ou n'importe où d'accessible
depuis une autre machine, faites **d'abord deux choses** :

1. Définissez `ROITELET_API_TOKEN` à une valeur non vide (verrouille
   `/api/chat`, `/api/settings`, `/api/conversations`,
   `/api/telemetry`, `/api/personal*`, `/api/images`, et
   `/v1/chat/completions`).
2. Soit conservez le service derrière un reverse proxy qui gère
   l'authentification, soit acceptez que le jeton est votre seule
   ligne de défense. Si vous basculez
   `ROITELET_APP_HOST=0.0.0.0` pour un `./start.sh` bare-metal, le
   LAN vous voit dès qu'uvicorn écoute.

Sans ces étapes, quiconque peut joindre le port peut lire vos
conversations, votre télémétrie brute (qui contient les prompts et
les réponses des fournisseurs), et déclencher des appels payants
sur vos clés API. Modèle de menace détaillé :
[docs/PRIVACY.md](docs/PRIVACY.md).

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
├── docs/ARCHITECTURE.md        # Architecture détaillée (Mermaid) — contributeurs
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
  projection 2-D des embeddings (style Karpathy) et l'index RAG
  persistant accéléré par turbovec (`pip install -e .[personal]`).
- **[docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md)** — sélection
  de route (`/image`, `/speech`, `/personal`, `/help`) et matrice
  associant les préférences par tour aux contrôles visibles
  (curseurs GUI / arguments CLI / booléens API).
- **[PSEUDO.md](docs/PSEUDO.md)** — pseudonymisation : taxonomie PII,
  contrat fail-closed, surfaces GUI / CLI / API, audit.
- **[docs/PRIVACY.md](docs/PRIVACY.md)** — local-first vs local-only,
  ce qui est stocké sur disque, ce qui part sur le réseau.
- **[docs/EVALUATION.md](docs/EVALUATION.md)** — feuille de route
  permanente des ablations : quelles configurations comparer,
  quelles métriques, ce qui a été exécuté, ce qui reste planifié.

### Niveau 3 — Contributeurs (vous voulez *modifier* Roitelet)
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — visite architecturale complète
  avec diagrammes Mermaid. Maths de routage, régimes, boucle Elo,
  les deux routeurs, les deux détecteurs de capacité, pipeline
  image-gen.

---

## Licence

Distribué sous **licence BSD 3-Clause** — voir [LICENSE](LICENSE).

## Auteur

[Warith HARCHAOUI](https://www.linkedin.com/in/warith-harchaoui/)
