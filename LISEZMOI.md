# Roitelet LLM

> **Posez une question — plusieurs IA répondent en même temps — un
> petit modèle sur votre ordinateur sélectionne les meilleurs morceaux
> de chaque réponse et vous donne une seule réponse.**

Roitelet tourne sur votre machine. Vous pouvez utiliser l'IA de votre
ordinateur portable, ou brancher des modèles cloud (ChatGPT, Claude,
Gemini via OpenRouter, …) et les faire concourir sur chaque question.
En option, Roitelet masque vos informations personnelles avant que
quoi que ce soit parte vers le cloud et les remet dans la réponse.

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

---

## Quel document devrais-je lire ?

Choisissez la ligne qui correspond à ce que vous êtes venu faire.

| Vous êtes… | …et vous voulez | Commencez par |
|---|---|---|
| 🧑 **Un·e utilisateur·rice curieux·se** | Essayer Roitelet sur votre ordinateur, lui poser une question | [Démarrage rapide](#démarrage-rapide) (ci-dessous) |
| 🧰 **Un·e utilisateur·rice avec des fichiers** | Déposer PDF, audio, images, ou une URL et interroger | [docs/PERSONAL_MODE.md](docs/PERSONAL_MODE.md) |
| 🔐 **Un·e utilisateur·rice attentif·ve à la vie privée** | Comprendre ce qui reste local et comment fonctionne le masquage des PII | [docs/PRIVACY.md](docs/PRIVACY.md) → [docs/PSEUDO.md](docs/PSEUDO.md) |
| 🧑‍💻 **Un·e dev avec un outillage OpenAI existant** | Brancher votre SDK `openai` / LiteLLM / Continue.dev sur Roitelet | [docs/OPENAI_COMPAT.md](docs/OPENAI_COMPAT.md) |
| 🏗️ **Un·e dev qui ajoute des modèles** | Brancher un GGUF local, OpenAI, Mistral, Together, etc. | [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) |
| 🎛️ **Un·e utilisateur·rice avancé·e** | Utiliser les routes slash (`/image`, `/personal`, …) et contrôles par tour | [docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md) |
| 🖥️ **Un·e sysadmin / installateur·rice** | Installer sur Linux/Mac/Windows avec conda, venv ou Docker | [INSTALLER.md](INSTALLER.md) ([English](INSTALL.md)) |
| 🔬 **Un·e chercheur·se / sceptique honnête** | Voir les chiffres — la fusion aide-t-elle vraiment ? | [docs/EVALUATION.md](docs/EVALUATION.md) |
| 🛠️ **Un·e contributeur·rice / forkeur·se** | Comprendre l'intérieur : routeur, régimes, boucle Elo | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 🇬🇧 **Anglophone** | Tout ce qui précède, en anglais | [README.md](README.md) |

---

## Démarrage rapide

Cinq minutes pour avoir Roitelet sur votre machine :

```bash
# 1. Installer Ollama (une fois).
#    macOS : brew install ollama
#    Linux : curl -fsSL https://ollama.com/install.sh | sh

# 2. Télécharger un petit modèle local pour donner à Roitelet quelqu'un à qui parler.
ollama pull qwen3:8b
ollama pull nomic-embed-text     # minuscule — utilisé par le mode personnel

# 3. Installer Roitelet.
git clone https://github.com/<votre-fork>/roitelet-llm.git
cd roitelet-llm
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 4. Lancer.
./start.sh                       # ouvre http://localhost:8000
```

Ouvrez `http://localhost:8000` et posez n'importe quelle question.
L'interface web s'explique d'elle-même — bascule de langue (EN/FR)
dans l'en-tête de la barre latérale, icône « curseurs » à côté du
bouton d'envoi pour les options par message, et une feuille Réglages
derrière l'engrenage en bas de la barre latérale.

Vous préférez le terminal ? Mêmes opérations, surface différente.
Les commentaires `#` décrivent **le type de sortie** que vous verrez —
le texte réel dépend des modèles que vous avez configurés.

```bash
roitelet ask "Explique le tri rapide en un paragraphe."
# → affiche un paragraphe markdown : la réponse fusionnée par le juge
#   à partir de vos top-K modèles. Pas de métadonnées sauf si --verbose.

roitelet ask --pseudonymize "Email à Marie Dupont à marie@orange.fr au sujet du Q3."
#
# Ce qui se passe, étape par étape :
#
# 1. Le pseudonymiseur local réécrit le prompt avant le fan-out — chaque
#    nom de personne et coordonnée reçoit un substitut plausible de même
#    origine :
#       prompt qui sort du pseudonymiseur local →
#       "Email à Camille Lefèvre à camille.lefevre@orange.fr au sujet du Q3."
#
# 2. Les modèles candidats (locaux ou distants, selon ce que le routeur
#    a choisi) répondent à ce prompt réécrit. Ils ne voient jamais
#    « Marie Dupont » ni l'email réel. Si un fournisseur cloud journalise
#    la requête, c'est cela qu'il journalise.
#
# 3. La passe inverse locale remet les originaux dans la réponse fusionnée,
#    donc l'utilisateur voit :
#       Objet : Mise à jour Q3
#       Chère Marie Dupont,
#       J'aimerais partager les chiffres du Q3 avec vous …
#
# --verbose montre l'audit complet (chaque paire original → substitut,
# le prompt exact envoyé, les latences forward + reverse).

roitelet ask --url https://docs.python.org/3/library/asyncio.html "Résume."
# → Firecrawl scrape la page localement (ou via votre FIRECRAWL_API_KEY),
#   préfixe le markdown comme bloc [Website: …], puis lance la pipeline
#   normale. La synthèse est votre résumé fusionné.

roitelet chat --independence    # REPL interactif, en local uniquement
# → un prompt « You> » ; tapez votre question, appuyez sur Entrée,
#   « Roitelet> » affiche la réponse fusionnée. Aucun candidat distant
#   n'est appelé. Tapez « exit » ou Ctrl-D pour sortir.

roitelet settings get           # voir ce qui est persisté
# → affiche le AppSettingsPayload complet en JSON formaté : identifiants
#   de modèles, URL Ollama, clés API masquées, poids écofrugalité, etc.
#   Utilisez `roitelet settings get <key>` pour lire un seul champ.
```

Pour les détails d'installation (Docker, bundles de modèles,
comparaison de profils) : [INSTALLER.md](INSTALLER.md) (français),
[INSTALL.md](INSTALL.md) (anglais).

---

## Ce que ça fait (en clair)

| Fonctionnalité | Ce que ça veut dire pour vous |
|---|---|
| **Comparer les modèles** | Une question, plusieurs réponses (par ex. Claude + Llama + Gemma), une réponse finale. |
| **Local-first** | Si vous ne configurez que des modèles locaux, **rien ne quitte votre machine**. |
| **Masquer les infos perso** | Activez « Pseudonymiser » — noms, adresses, identifiants sont remplacés par des fakes plausibles avant l'envoi, restaurés dans la réponse. |
| **Joindre des fichiers** | Audio (transcrit), images (lues par un modèle vision), PDF (texte extrait) — tout localement. |
| **Joindre des sites web** | Collez une URL — Roitelet scrape la page (Firecrawl) et l'inclut dans le prompt. |
| **RAG personnel** | Déposez vos propres notes dans un dossier ; Roitelet s'en sert pour répondre. |
| **Génération d'images** | Si vous avez configuré DALL-E / Stable Diffusion / Imagen, demandez avec `/image`. |
| **Mêmes opérations en CLI et API** | Tout ce que vous faites dans l'interface, vous le faites en terminal ou par appel HTTP. |

---

## Comment ça marche (un diagramme)

```mermaid
flowchart LR
    U[Prompt utilisateur] --> P{Pseudonymiser ?<br>(opt-in)}
    P -- oui --> PFW[LLM local<br>retire les PII] --> R
    P -- non --> R
    R[Routeur<br>priors capacités<br>+ Elo glissant<br>+ régimes] --> SEL[Top-K<br>candidats]
    SEL -.parallèle.-> C1[Candidat 1]
    SEL -.parallèle.-> C2[Candidat 2]
    SEL -.parallèle.-> CN[Candidat K]
    C1 --> J[Juge local<br>anonymisé<br>+ mélangé]
    C2 --> J
    CN --> J
    J --> REV{Pseudo activé ?}
    REV -- oui --> PREV[LLM local<br>restaure les PII] --> A
    REV -- non --> A
    A[Réponse fusionnée] --> USER[Utilisateur]
    J -.gagnants.-> ELO[(Elo glissant<br>par capacité)]
    ELO -.tour suivant.-> R
    style P fill:#fef3c7,stroke:#f59e0b
    style REV fill:#fef3c7,stroke:#f59e0b
    style PFW fill:#fef3c7,stroke:#f59e0b
    style PREV fill:#fef3c7,stroke:#f59e0b
    style J fill:#dbeafe,stroke:#3b82f6
    style ELO fill:#f3e8ff,stroke:#a855f7
```

Par tour :

1. **Routeur** sélectionne les top-K modèles pour la question (K=2 par défaut).
2. **Fan-out** — les K modèles répondent en parallèle.
3. **Juge** — un modèle local lit les K réponses (anonymisées et
   mélangées) et les fusionne en une seule.
4. **Mise à jour Elo** — les gagnants gagnent des points sur le sujet ;
   les perdants en perdent. Le tour suivant profite de ce signal.

Les détails internes (les maths, les détecteurs de régime, la
variante du routeur en factorisation matricielle) vivent dans
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Trois surfaces, mêmes fonctionnalités

| Surface | Accès | Préférences par tour |
|---|---|---|
| **Web** (GUI) | `http://localhost:8000/` après `./start.sh` | Icône curseurs à côté du bouton d'envoi |
| **CLI** | `roitelet ask "…"` / `roitelet chat` | `--top-k`, `--independence`, `--pseudonymize`, `--max-cost-usd`, `--quality-threshold`, `--url[ --url-recursive]`, `--verbose` |
| **API (MCP inclus)** | `POST /api/chat` natif, `POST /v1/chat/completions` compatible OpenAI, MCP JSON-RPC `POST /mcp` (Model Context Protocol — fonctionne comme source d'outils pour Claude Desktop, Cursor, et tout client MCP) | `preferences.{independence, pseudonymize, top_k, max_cost_usd, quality_threshold}` dans le body JSON ; pour MCP, des champs de même nom sur l'appel d'outil `roitelet.chat`. |

Pour les clients OpenAI : voir [docs/OPENAI_COMPAT.md](docs/OPENAI_COMPAT.md).

---

## Latence, coût, quand ne **pas** utiliser

Le wall-clock d'un tour est `max(latences_candidats) + latence_juge`
parce que le fan-out passe par `asyncio.gather`. Les modèles locaux
sont gratuits au token marginal mais coûtent en RAM/VRAM. Les
candidats distants coûtent ce que leur fournisseur facture.

K=2 est le point d'équilibre empirique sur le dataset (voir
[docs/EVALUATION.md](docs/EVALUATION.md)).

**Quand ne pas utiliser Roitelet :**

- **UX chat très basse latence.** Un seul modèle rapide bat le
  fan-out + fusion.
- **Prompts triviaux.** « 2+2 ? » n'a pas besoin de trois opinions.
- **Trafic production gros volume.** Le coût est multiplicatif.
- **Vous voulez juste une passerelle.** C'est le boulot de
  [LiteLLM](https://github.com/BerriAI/litellm).

---

## Note sécurité

Roitelet livre **sûr par défaut** : `start.sh` se bind à `127.0.0.1`
et `ROITELET_API_TOKEN` est vide. Localhost-only sans auth est OK
pour un laptop mono-utilisateur.

**Avant d'exposer sur LAN, Internet, ngrok, Tailscale, etc. :**

1. Mettez `ROITELET_API_TOKEN` à une valeur non vide.
2. Soit gardez le service derrière un reverse proxy qui gère l'auth,
   soit acceptez que le token soit votre seule ligne de défense.

Modèle de menace : [docs/PRIVACY.md](docs/PRIVACY.md).

---

## Comment Roitelet diffère des projets voisins

| Projet | Rôle principal | Comment Roitelet diffère |
|---|---|---|
| [LiteLLM](https://github.com/BerriAI/litellm) | Passerelle compatible OpenAI sur plusieurs APIs | Roitelet est plus étroit : fan-out local-first + synthèse locale + Elo inspectable. LiteLLM est un candidat que Roitelet pourrait appeler. |
| [OpenRouter](https://openrouter.ai) | Marketplace multi-modèles hébergé | Roitelet tourne sur votre machine. OpenRouter est un fournisseur de candidats, pas un remplaçant. |
| [RouteLLM](https://github.com/lm-sys/RouteLLM) | Routage coût-conscient strong/weak entraîné sur préférences | Roitelet fait top-K + fusion, pas du routage binaire. Roitelet expose un bouton `quality_threshold` de même forme (scalaire unique, monotone), dérivé de l'Elo glissant. |
| [LangChain](https://www.langchain.com) / LangGraph | Frameworks d'orchestration LLM | Roitelet est un système end-user, pas un framework. |
| Clients chat mono-modèle | Un modèle entre, une réponse sort | Roitelet échange simplicité et latence contre comparaison + redondance + synthèse. |

---

## Licence

Publié sous la **licence BSD à 3 clauses** — voir [LICENSE](LICENSE).

## Auteur

[Warith HARCHAOUI](https://www.linkedin.com/in/warith-harchaoui/)
