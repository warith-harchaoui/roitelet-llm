/* Roitelet i18n — EN / FR, plain-language strings.
 *
 * Design notes for the strings themselves (read this before adding a key):
 *
 * 1. Plain language wins. Most users have never heard "top-K", "fan-out",
 *    "ecofrugality", or "PII". The visible label is what a non-technical
 *    user would type into a search box — "How many models answer at once?",
 *    not "Top-K fan-out". The technical term lives in the help text or
 *    tooltip.
 * 2. Verbs in the imperative for buttons ("Send", "Save", "New chat").
 *    Nouns for labels ("Settings", "Conversation").
 * 3. Errors describe what to do next, not what went wrong internally.
 *    "Couldn't reach a model" is more useful than "Pipeline error: 502".
 * 4. French strings are translations of *meaning*, not word-for-word.
 *    "Settings" → "Réglages" (not "Configurations"), "New chat" →
 *    "Nouvelle conversation".
 *
 * Keys are namespaced with dots so the search-and-replace of one panel
 * doesn't risk colliding with another. The `t(key, vars)` helper does
 * ``{name}`` substitution.
 */

const TRANSLATIONS = {
  en: {
    // Sidebar
    'sidebar.newChat': 'New chat',
    'sidebar.newChat.tooltip': 'Start a new conversation — ⌘N',
    'sidebar.settings': 'Settings',
    'sidebar.language.toggle': 'Switch to French',
    'sidebar.openMenu': 'Open menu',

    // Header
    'header.status.ready': 'Ready',
    'header.status.thinking': 'Thinking…',
    'header.status.loading': 'Loading…',
    'header.status.tag.pseudo': 'hiding personal info',
    'header.status.tag.local': 'offline mode',

    // Welcome
    'welcome.title': 'Ask anything.',
    'welcome.subtitle': 'Several AI models answer your question at the same time, and a local model picks the best parts of each answer for you.',
    'welcome.tip': 'Tip — start a message with <code>/image</code> to draw, <code>/personal</code> to use your private files, or <code>/help</code> for the list. For other settings use the sliders icon below.',

    // Composer
    'composer.placeholder': 'Ask Roitelet anything…',
    'composer.send': 'Send',
    'composer.attach': 'Attach a file (audio, image, or PDF)',
    'composer.prefs.tooltip': 'Choose what to do for this message only',
    'composer.shortcuts': '<kbd class="kbd">⌘↵</kbd> to send · <kbd class="kbd">⌘N</kbd> for a new chat',

    // Per-turn preferences popover
    'prefs.title': 'For this message only',
    'prefs.independence': 'Stay offline (only models on my computer)',
    'prefs.pseudonymize': 'Hide my personal info before sending',
    'prefs.topK': 'How many AIs should answer',
    'prefs.maxCost': 'Spending limit (US dollars)',
    'prefs.maxCost.placeholder': 'no limit',
    'prefs.footer': 'Defaults live in <button type="button" id="prefsToSettings" class="text-sysblue hover:underline">Settings</button>.',

    // Settings sheet
    'settings.title': 'Settings',
    'settings.cancel': 'Cancel',
    'settings.save': 'Save',
    'settings.close': 'Close settings',
    'settings.section.basics': 'Basics',
    'settings.section.advanced': 'Advanced',
    'settings.section.advanced.hint': 'Power-user knobs. Leave the defaults if you are unsure.',

    // Personal panel
    'personal.title': 'My private files',
    'personal.subtitle.loading': 'Loading…',
    'personal.subtitle.summary': '{wiki} note(s) in your wiki · {inbox} new file(s) to ingest · mode={mode}',
    'personal.ingest': 'Ingest new files',
    'personal.ingesting': 'Reading files…',
    'personal.visualize': 'Show map',
    'personal.help': 'Drop audio, images, PDFs, or Markdown into <code>data/personal/inbox/</code>. Click <em>Ingest new files</em> to turn them into searchable notes. Then start a message with <code>/personal &lt;your question&gt;</code> to query.',
    'personal.ingested': 'Read {added} file(s). Your wiki now has {wiki} note(s).',
    'personal.ingest.failed': 'Could not ingest: {message}',
    'personal.viz.empty': 'No notes to map yet, or the embedding model is unavailable.',
    'personal.viz.fetchFailed': 'Could not fetch the map: {message}',
    'personal.viz.title': 'My notes — semantic map',
    'personal.viz.meta': '{points} chunks across {sources} file(s) — hover a dot to see the excerpt.',
    'personal.viz.close': 'Close',

    // Engines panel
    'engines.title': 'Cloud AI providers',
    'engines.subtitle': 'Add any provider with an OpenAI-style endpoint. Examples: Mistral, Together, Groq, Fireworks, llama-server.',
    'engines.addEngine': '+ Add provider',
    'engines.empty': 'No providers added yet. Click <strong>+ Add provider</strong>. Common ones: {presets}.',
    'engines.label.placeholder': 'label (e.g. mistral)',
    'engines.baseUrl.placeholder': 'https://api.example.com/v1',
    'engines.apiKey.placeholder': 'api key',
    'engines.models.placeholder': 'models (comma-separated)',
    'engines.remove': 'Remove this provider',

    // Settings fields — plain-language labels.
    // The key is the AppSettingsPayload field name; the value is the
    // visible label. Fields not in this map fall back to the raw key.
    'field.ollama_base_url': 'Local AI (Ollama) server URL',
    'field.local_synthesis_model': 'Local Judge',
    'field.local_vlm_model': 'Local model that reads images',
    'field.localModels.empty': 'No local models found. Install one with `ollama pull qwen3:8b` then refresh.',
    'field.selected_ollama_models': 'Other local models to use (comma-separated)',
    'field.openrouter_api_key': 'OpenRouter API key',
    'field.paid_openrouter_models': 'OpenRouter models (comma-separated)',
    'field.openai_api_key': 'OpenAI API key',
    'field.anthropic_api_key': 'Anthropic API key',
    'field.gemini_api_key': 'Google Gemini API key',
    'field.perplexity_api_key': 'Perplexity API key',
    'field.raw_power_weight': 'Weight for answer quality (0–1)',
    'field.ecofrugality_weight': 'Weight for low cost + low energy (0–1)',
    'field.independence_local_only': 'Default to offline mode (only my local models)',
    'field.enable_vlms': 'Allow models to read images',
    'field.enable_pseudonymization': 'Hide my personal info before sending (default)',
    'field.pseudo_model_id': 'Model that hides personal info (blank = same as picker)',

    // Audit affordance under the user bubble
    'audit.summary': 'Personal info hidden · {count} item · view what was sent',
    'audit.summary.plural': 'Personal info hidden · {count} items · view what was sent',
    'audit.summary.zero': 'Personal info checked — nothing to hide · view',
    'audit.sentLabel': 'What the cloud models actually saw',
    'audit.tableLabel': 'What was hidden',
    'audit.empty': 'No personal info was found — your message was sent as-is.',
    'audit.timing': '{model} · prepared in {fwd}s · restored in {rev}s · {repair}',
    'audit.repair.used': 'used a second pass to handle inflected names',
    'audit.repair.skipped': 'one-pass restore',

    // Bot metadata details (the existing collapsible)
    'meta.capabilities': 'Topic',
    'meta.latency': 'Time',

    // Toasts / errors
    'toast.saved': 'Saved',
    'toast.saveFailed': 'Couldn\'t save: {message}',
    'toast.loadFailed': 'Couldn\'t load conversations: {message}',
    'toast.convoFailed': 'Couldn\'t load this conversation: {message}',
    'toast.settingsLoadFailed': 'Couldn\'t load settings: {message}',
    'toast.pipelineError': 'Something went wrong: {message}',
    'toast.skippedAttachment': '{name} was skipped — only audio, images, or PDFs work.',
    'toast.skippedVision': '{name} was skipped — turn on "Allow models to read images" in Settings first.',
    'toast.imageNeedsPrompt': '"/image" needs words after it — try "/image a wren in a forest".',
    'toast.speechNeedsAudio': '"/speech" needs an audio file. Click the paperclip first.',
    'toast.imageGenerated': 'Created {n} image(s) with {model}.',

    // Status footer
    'misc.untitled': 'Untitled',
    'misc.noAnswer': '(no answer)',
    'misc.attachmentOnly': '(attachment only)',
  },

  fr: {
    'sidebar.newChat': 'Nouvelle conversation',
    'sidebar.newChat.tooltip': 'Démarrer une nouvelle conversation — ⌘N',
    'sidebar.settings': 'Réglages',
    'sidebar.language.toggle': 'Passer en anglais',
    'sidebar.openMenu': 'Ouvrir le menu',

    'header.status.ready': 'Prêt',
    'header.status.thinking': 'Je réfléchis…',
    'header.status.loading': 'Chargement…',
    'header.status.tag.pseudo': 'infos perso masquées',
    'header.status.tag.local': 'mode hors-ligne',

    'welcome.title': 'Posez n\'importe quelle question.',
    'welcome.subtitle': 'Plusieurs IA répondent en même temps et un modèle local sélectionne les meilleurs morceaux de chaque réponse pour vous.',
    'welcome.tip': 'Astuce — commencez par <code>/image</code> pour dessiner, <code>/personal</code> pour interroger vos fichiers privés, ou <code>/help</code> pour la liste. Pour les autres réglages, utilisez l\'icône curseurs ci-dessous.',

    'composer.placeholder': 'Demandez n\'importe quoi à Roitelet…',
    'composer.send': 'Envoyer',
    'composer.attach': 'Joindre un fichier (audio, image ou PDF)',
    'composer.prefs.tooltip': 'Choisir ce qu\'on fait pour ce message uniquement',
    'composer.shortcuts': '<kbd class="kbd">⌘↵</kbd> pour envoyer · <kbd class="kbd">⌘N</kbd> pour une nouvelle conversation',

    'prefs.title': 'Pour ce message uniquement',
    'prefs.independence': 'Rester hors-ligne (uniquement les modèles sur mon ordi)',
    'prefs.pseudonymize': 'Masquer mes infos perso avant l\'envoi',
    'prefs.topK': 'Combien d\'IA répondent',
    'prefs.maxCost': 'Limite de dépense (dollars US)',
    'prefs.maxCost.placeholder': 'aucune limite',
    'prefs.footer': 'Les valeurs par défaut sont dans <button type="button" id="prefsToSettings" class="text-sysblue hover:underline">Réglages</button>.',

    'settings.title': 'Réglages',
    'settings.cancel': 'Annuler',
    'settings.save': 'Enregistrer',
    'settings.close': 'Fermer les réglages',
    'settings.section.basics': 'Essentiels',
    'settings.section.advanced': 'Avancés',
    'settings.section.advanced.hint': 'Réglages pour utilisateurs avertis. Laissez les valeurs par défaut si vous hésitez.',

    'personal.title': 'Mes fichiers privés',
    'personal.subtitle.loading': 'Chargement…',
    'personal.subtitle.summary': '{wiki} note(s) dans le wiki · {inbox} nouveau(x) fichier(s) à ingérer · mode={mode}',
    'personal.ingest': 'Lire les nouveaux fichiers',
    'personal.ingesting': 'Lecture des fichiers…',
    'personal.visualize': 'Afficher la carte',
    'personal.help': 'Déposez audio, images, PDF ou Markdown dans <code>data/personal/inbox/</code>. Cliquez <em>Lire les nouveaux fichiers</em> pour les transformer en notes consultables. Puis commencez un message par <code>/personal &lt;votre question&gt;</code> pour interroger.',
    'personal.ingested': '{added} fichier(s) lu(s). Votre wiki contient maintenant {wiki} note(s).',
    'personal.ingest.failed': 'Impossible d\'ingérer : {message}',
    'personal.viz.empty': 'Aucune note à cartographier pour le moment, ou le modèle d\'embedding n\'est pas joignable.',
    'personal.viz.fetchFailed': 'Impossible de récupérer la carte : {message}',
    'personal.viz.title': 'Mes notes — carte sémantique',
    'personal.viz.meta': '{points} morceaux répartis sur {sources} fichier(s) — passez la souris sur un point pour voir l\'extrait.',
    'personal.viz.close': 'Fermer',

    'engines.title': 'Fournisseurs d\'IA cloud',
    'engines.subtitle': 'Ajoutez n\'importe quel fournisseur avec un endpoint style OpenAI. Exemples : Mistral, Together, Groq, Fireworks, llama-server.',
    'engines.addEngine': '+ Ajouter un fournisseur',
    'engines.empty': 'Aucun fournisseur ajouté pour l\'instant. Cliquez <strong>+ Ajouter un fournisseur</strong>. Les plus courants : {presets}.',
    'engines.label.placeholder': 'étiquette (ex. mistral)',
    'engines.baseUrl.placeholder': 'https://api.exemple.com/v1',
    'engines.apiKey.placeholder': 'clé API',
    'engines.models.placeholder': 'modèles (séparés par des virgules)',
    'engines.remove': 'Retirer ce fournisseur',

    'field.ollama_base_url': 'URL du serveur IA local (Ollama)',
    'field.local_synthesis_model': 'Juge local',
    'field.local_vlm_model': 'Modèle local qui lit les images',
    'field.localModels.empty': 'Aucun modèle local trouvé. Installez-en un avec `ollama pull qwen3:8b` puis rafraîchissez.',
    'field.selected_ollama_models': 'Autres modèles locaux à utiliser (séparés par des virgules)',
    'field.openrouter_api_key': 'Clé API OpenRouter',
    'field.paid_openrouter_models': 'Modèles OpenRouter (séparés par des virgules)',
    'field.openai_api_key': 'Clé API OpenAI',
    'field.anthropic_api_key': 'Clé API Anthropic',
    'field.gemini_api_key': 'Clé API Google Gemini',
    'field.perplexity_api_key': 'Clé API Perplexity',
    'field.raw_power_weight': 'Poids pour la qualité de la réponse (0–1)',
    'field.ecofrugality_weight': 'Poids pour bas coût + basse énergie (0–1)',
    'field.independence_local_only': 'Mode hors-ligne par défaut (uniquement mes modèles locaux)',
    'field.enable_vlms': 'Autoriser les modèles à lire les images',
    'field.enable_pseudonymization': 'Masquer mes infos perso avant l\'envoi (par défaut)',
    'field.pseudo_model_id': 'Modèle qui masque les infos perso (vide = même que le sélecteur)',

    'audit.summary': 'Infos perso masquées · {count} élément · voir ce qui a été envoyé',
    'audit.summary.plural': 'Infos perso masquées · {count} éléments · voir ce qui a été envoyé',
    'audit.summary.zero': 'Infos perso vérifiées — rien à masquer · voir',
    'audit.sentLabel': 'Ce que les modèles cloud ont vraiment vu',
    'audit.tableLabel': 'Ce qui a été masqué',
    'audit.empty': 'Aucune info perso détectée — votre message a été envoyé tel quel.',
    'audit.timing': '{model} · préparé en {fwd}s · restauré en {rev}s · {repair}',
    'audit.repair.used': 'deuxième passe utilisée pour les noms infléchis',
    'audit.repair.skipped': 'restauration en une passe',

    'meta.capabilities': 'Sujet',
    'meta.latency': 'Temps',

    'toast.saved': 'Enregistré',
    'toast.saveFailed': 'Impossible d\'enregistrer : {message}',
    'toast.loadFailed': 'Impossible de charger les conversations : {message}',
    'toast.convoFailed': 'Impossible de charger cette conversation : {message}',
    'toast.settingsLoadFailed': 'Impossible de charger les réglages : {message}',
    'toast.pipelineError': 'Un problème est survenu : {message}',
    'toast.skippedAttachment': '{name} a été ignoré — seuls les audio, images ou PDF fonctionnent.',
    'toast.skippedVision': '{name} a été ignoré — activez d\'abord "Autoriser les modèles à lire les images" dans Réglages.',
    'toast.imageNeedsPrompt': '"/image" doit être suivi de mots — essayez "/image un roitelet dans une forêt".',
    'toast.speechNeedsAudio': '"/speech" nécessite un fichier audio. Cliquez d\'abord sur le trombone.',
    'toast.imageGenerated': '{n} image(s) créée(s) avec {model}.',

    'misc.untitled': 'Sans titre',
    'misc.noAnswer': '(pas de réponse)',
    'misc.attachmentOnly': '(pièce jointe uniquement)',
  },
};

// ─── i18n state + helper ────────────────────────────────────────────────────

const I18N_STORAGE_KEY = 'roitelet.lang';
let _currentLang = _initialLang();

function _initialLang() {
  // Priority: explicit user choice in localStorage > browser language
  // starts with "fr" > English.
  try {
    const saved = localStorage.getItem(I18N_STORAGE_KEY);
    if (saved === 'fr' || saved === 'en') return saved;
  } catch { /* localStorage unavailable, fall through */ }
  const nav = (navigator.language || '').toLowerCase();
  return nav.startsWith('fr') ? 'fr' : 'en';
}

function t(key, vars) {
  const bundle = TRANSLATIONS[_currentLang] || TRANSLATIONS.en;
  let text = bundle[key];
  if (text === undefined) {
    // Fall back to English so a missing FR translation never blanks the UI.
    text = TRANSLATIONS.en[key] !== undefined ? TRANSLATIONS.en[key] : key;
  }
  if (vars) {
    for (const k of Object.keys(vars)) {
      text = text.replaceAll('{' + k + '}', String(vars[k]));
    }
  }
  return text;
}

function currentLang() { return _currentLang; }

function setLang(lang) {
  if (lang !== 'en' && lang !== 'fr') return;
  _currentLang = lang;
  try { localStorage.setItem(I18N_STORAGE_KEY, lang); } catch { /* ignore */ }
  applyStaticTranslations();
  document.documentElement.setAttribute('lang', lang);
}

// Walk every element annotated with data-i18n / data-i18n-html /
// data-i18n-placeholder / data-i18n-title and rewrite it. ``-html``
// permits the small amount of inline markup (kbd, code, button) that
// the welcome / shortcuts / footer strings use.
function applyStaticTranslations() {
  for (const el of document.querySelectorAll('[data-i18n]')) {
    el.textContent = t(el.getAttribute('data-i18n'));
  }
  for (const el of document.querySelectorAll('[data-i18n-html]')) {
    el.innerHTML = t(el.getAttribute('data-i18n-html'));
  }
  for (const el of document.querySelectorAll('[data-i18n-placeholder]')) {
    el.setAttribute('placeholder', t(el.getAttribute('data-i18n-placeholder')));
  }
  for (const el of document.querySelectorAll('[data-i18n-title]')) {
    const key = el.getAttribute('data-i18n-title');
    el.setAttribute('title', t(key));
    if (el.hasAttribute('aria-label')) {
      el.setAttribute('aria-label', t(key));
    }
  }
  // Language toggle button itself.
  const langBtn = document.getElementById('langToggle');
  if (langBtn) {
    langBtn.textContent = _currentLang === 'fr' ? 'EN' : 'FR';
    langBtn.title = t('sidebar.language.toggle');
  }
}

window.RoiteletI18n = { t, currentLang, setLang, applyStaticTranslations };
