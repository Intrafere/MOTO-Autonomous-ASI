"""Target-language catalog and name matching for Auto Top-p Translate.

The catalog lists the standard languages offered in translation settings, with
six pinned languages first. Model-reported language names (from the
proficiency probe) are matched against English names, endonyms, and common
aliases. Custom languages get a deterministic key derived from their
normalized name, so the same name always resolves to the same stored
translations. Everything here is pure and safe to call from any thread.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence


CUSTOM_LANGUAGE_KEY_PREFIX = "custom-"
CUSTOM_LANGUAGE_KEY_HASH_CHARS = 10
CUSTOM_LANGUAGE_NAME_MAX_CHARS = 60


@dataclass(frozen=True)
class LanguageEntry:
    """One target language as offered in settings and named in prompts."""

    key: str
    name: str
    native_name: str
    aliases: tuple[str, ...] = ()
    descriptor: str = ""
    pinned: bool = False


def _lang(
    key: str,
    name: str,
    native_name: str = "",
    *aliases: str,
    descriptor: str = "",
    pinned: bool = False,
) -> LanguageEntry:
    return LanguageEntry(
        key=key,
        name=name,
        native_name=native_name,
        aliases=tuple(aliases),
        descriptor=descriptor,
        pinned=pinned,
    )


PINNED_LANGUAGE_KEYS: tuple[str, ...] = ("en", "zh-Hans", "hi", "es", "ar", "fr")

_PINNED_LANGUAGES = (
    _lang(
        "en", "English", "",
        "American English", "British English", "English (US)", "English (UK)",
        pinned=True,
    ),
    _lang(
        "zh-Hans", "Mandarin Chinese", "简体中文",
        "Chinese", "Chinese (Simplified)", "Simplified Chinese", "Mandarin",
        "Chinese Mandarin", "Standard Chinese", "Putonghua", "中文", "普通话",
        descriptor="Mandarin Chinese, written in Simplified Chinese characters",
        pinned=True,
    ),
    _lang("hi", "Hindi", "हिन्दी", pinned=True),
    _lang(
        "es", "Spanish", "Español",
        "Castilian", "European Spanish", "Latin American Spanish",
        "Spanish (Spain)", "Spanish (Latin America)",
        pinned=True,
    ),
    _lang(
        "ar", "Modern Standard Arabic", "العربية الفصحى",
        "Arabic", "Standard Arabic", "Literary Arabic", "MSA", "Fusha", "العربية",
        pinned=True,
    ),
    _lang("fr", "French", "Français", "French (France)", "Standard French", pinned=True),
)

# Authored roughly alphabetically; the public tuple sorts them by English name.
# Descriptors pin one script for languages commonly written in several, so every
# passage of a document comes back in the same script.
_OTHER_LANGUAGES = (
    _lang("af", "Afrikaans"),
    _lang("ak", "Akan", "", "Twi", "Asante Twi"),
    _lang("sq", "Albanian", "Shqip"),
    _lang("am", "Amharic", "አማርኛ"),
    _lang("hy", "Armenian", "Հայերեն"),
    _lang("as", "Assamese", "অসমীয়া"),
    _lang("ay", "Aymara", "Aymar aru"),
    _lang(
        "az", "Azerbaijani", "Azərbaycan dili", "Azeri", "North Azerbaijani",
        descriptor="Azerbaijani, written in Latin script",
    ),
    _lang("eu", "Basque", "Euskara"),
    _lang("be", "Belarusian", "Беларуская"),
    _lang("bn", "Bengali", "বাংলা", "Bangla"),
    _lang("bho", "Bhojpuri", "भोजपुरी"),
    _lang("bs", "Bosnian", "Bosanski", descriptor="Bosnian, written in Latin script"),
    _lang("bg", "Bulgarian", "Български"),
    _lang("my", "Burmese", "မြန်မာဘာသာ", "Myanmar"),
    _lang(
        "zh-yue", "Cantonese", "粵語",
        "Yue Chinese", "Cantonese Chinese", "Chinese (Cantonese)", "廣東話",
        descriptor="Cantonese, written in Traditional Chinese characters",
    ),
    _lang("ca", "Catalan", "Català", "Valencian"),
    _lang("ceb", "Cebuano", "Sinugbuanon", "Bisaya", "Binisaya"),
    _lang("ny", "Chichewa", "", "Nyanja", "Chewa"),
    _lang(
        "zh-Hant", "Chinese (Traditional)", "繁體中文",
        "Traditional Chinese", "Taiwanese Mandarin", "Mandarin (Traditional)",
        descriptor="Mandarin Chinese, written in Traditional Chinese characters",
    ),
    _lang("co", "Corsican", "Corsu"),
    _lang("hr", "Croatian", "Hrvatski"),
    _lang("cs", "Czech", "Čeština"),
    _lang("da", "Danish", "Dansk"),
    _lang("dv", "Dhivehi", "ދިވެހި", "Divehi", "Maldivian"),
    _lang("nl", "Dutch", "Nederlands", "Flemish"),
    _lang("eo", "Esperanto"),
    _lang("et", "Estonian", "Eesti"),
    _lang("fil", "Filipino", "", "Tagalog"),
    _lang("fi", "Finnish", "Suomi"),
    _lang("gl", "Galician", "Galego"),
    _lang("ka", "Georgian", "ქართული"),
    _lang("de", "German", "Deutsch", "Standard German", "High German"),
    _lang("el", "Greek", "Ελληνικά", "Modern Greek"),
    _lang("gn", "Guarani", "Avañe'ẽ", "Guaraní", "Paraguayan Guarani"),
    _lang("gu", "Gujarati", "ગુજરાતી"),
    _lang("ht", "Haitian Creole", "Kreyòl ayisyen", "Haitian", "Kreyol"),
    _lang(
        "ha", "Hausa", "Harshen Hausa",
        descriptor="Hausa, written in Latin script (Boko)",
    ),
    _lang("haw", "Hawaiian", "ʻŌlelo Hawaiʻi"),
    _lang("he", "Hebrew", "עברית", "Modern Hebrew"),
    _lang("hmn", "Hmong", "Hmoob", "White Hmong", "Hmong Daw"),
    _lang("hu", "Hungarian", "Magyar"),
    _lang("is", "Icelandic", "Íslenska"),
    _lang("ig", "Igbo", "Asụsụ Igbo"),
    _lang("id", "Indonesian", "Bahasa Indonesia"),
    _lang("ga", "Irish", "Gaeilge", "Irish Gaelic"),
    _lang("it", "Italian", "Italiano"),
    _lang("ja", "Japanese", "日本語"),
    _lang("jv", "Javanese", "Basa Jawa"),
    _lang("kn", "Kannada", "ಕನ್ನಡ"),
    _lang(
        "kk", "Kazakh", "Қазақ тілі",
        descriptor="Kazakh, written in Cyrillic script",
    ),
    _lang("km", "Khmer", "ភាសាខ្មែរ", "Cambodian"),
    _lang("rw", "Kinyarwanda", "Ikinyarwanda"),
    _lang("ko", "Korean", "한국어"),
    _lang(
        "ku", "Kurdish (Kurmanji)", "Kurmancî", "Kurdish", "Kurmanji", "Northern Kurdish",
        descriptor="Kurdish (Kurmanji), written in Latin script",
    ),
    _lang(
        "ckb", "Kurdish (Sorani)", "سۆرانی", "Sorani", "Central Kurdish",
        descriptor="Kurdish (Sorani), written in Arabic script",
    ),
    _lang(
        "ky", "Kyrgyz", "Кыргызча", "Kirghiz", "Kirgiz",
        descriptor="Kyrgyz, written in Cyrillic script",
    ),
    _lang("lo", "Lao", "ລາວ", "Laotian"),
    _lang("la", "Latin", "Latina"),
    _lang("lv", "Latvian", "Latviešu"),
    _lang("ln", "Lingala", "Lingála"),
    _lang("lt", "Lithuanian", "Lietuvių"),
    _lang("lb", "Luxembourgish", "Lëtzebuergesch", "Letzeburgesch"),
    _lang("mk", "Macedonian", "Македонски"),
    _lang("mai", "Maithili", "मैथिली"),
    _lang("mg", "Malagasy"),
    _lang(
        "ms", "Malay", "Bahasa Melayu", "Malaysian", "Bahasa Malaysia",
        descriptor="Malay, written in Latin script",
    ),
    _lang("ml", "Malayalam", "മലയാളം"),
    _lang("mt", "Maltese", "Malti"),
    _lang("mi", "Maori", "Te Reo Māori", "Māori"),
    _lang("mr", "Marathi", "मराठी"),
    _lang(
        "mn", "Mongolian", "Монгол хэл",
        descriptor="Mongolian, written in Cyrillic script",
    ),
    _lang("ne", "Nepali", "नेपाली", "Nepalese"),
    _lang(
        "nb", "Norwegian (Bokmål)", "Norsk bokmål",
        "Norwegian", "Bokmål", "Bokmal", "Norwegian Bokmål", "Norwegian Bokmal",
    ),
    _lang("nn", "Norwegian (Nynorsk)", "Norsk nynorsk", "Nynorsk", "Norwegian Nynorsk"),
    _lang("or", "Odia", "ଓଡ଼ିଆ", "Oriya"),
    _lang("om", "Oromo", "Afaan Oromoo"),
    _lang("ps", "Pashto", "پښتو", "Pushto", "Pakhto"),
    _lang("fa", "Persian", "فارسی", "Farsi", "Iranian Persian"),
    _lang("pl", "Polish", "Polski"),
    _lang(
        "pt-BR", "Portuguese (Brazil)", "Português (Brasil)",
        "Portuguese", "Português", "Brazilian Portuguese", "Portuguese (Brazilian)",
    ),
    _lang(
        "pt-PT", "Portuguese (Portugal)", "Português (Portugal)",
        "Portuguese", "Português", "European Portuguese", "Portuguese (European)",
    ),
    _lang(
        "pa", "Punjabi", "ਪੰਜਾਬੀ", "Panjabi", "Punjabi (Gurmukhi)",
        descriptor="Punjabi, written in Gurmukhi script",
    ),
    _lang("qu", "Quechua", "Runa Simi"),
    _lang("ro", "Romanian", "Română"),
    _lang("ru", "Russian", "Русский"),
    _lang("sm", "Samoan", "Gagana Sāmoa"),
    _lang("gd", "Scottish Gaelic", "Gàidhlig", "Gaelic", "Scots Gaelic"),
    _lang(
        "sr", "Serbian", "Српски", "Srpski",
        descriptor="Serbian, written in Cyrillic script",
    ),
    _lang("st", "Sesotho", "", "Sotho", "Southern Sotho"),
    _lang("sn", "Shona", "chiShona"),
    _lang(
        "sd", "Sindhi", "سنڌي",
        descriptor="Sindhi, written in Perso-Arabic script",
    ),
    _lang("si", "Sinhala", "සිංහල", "Sinhalese"),
    _lang("sk", "Slovak", "Slovenčina"),
    _lang("sl", "Slovenian", "Slovenščina", "Slovene"),
    _lang("so", "Somali", "Soomaali"),
    _lang("su", "Sundanese", "Basa Sunda"),
    _lang("sw", "Swahili", "Kiswahili"),
    _lang("sv", "Swedish", "Svenska"),
    _lang(
        "tg", "Tajik", "Тоҷикӣ", "Tajiki",
        descriptor="Tajik, written in Cyrillic script",
    ),
    _lang("ta", "Tamil", "தமிழ்"),
    _lang(
        "tt", "Tatar", "Татар теле", "Volga Tatar",
        descriptor="Tatar, written in Cyrillic script",
    ),
    _lang("te", "Telugu", "తెలుగు"),
    _lang("th", "Thai", "ภาษาไทย"),
    _lang("bo", "Tibetan", "བོད་སྐད", "Standard Tibetan", "Lhasa Tibetan"),
    _lang("ti", "Tigrinya", "ትግርኛ"),
    _lang("tn", "Tswana", "Setswana"),
    _lang("tr", "Turkish", "Türkçe"),
    _lang(
        "tk", "Turkmen", "Türkmençe",
        descriptor="Turkmen, written in Latin script",
    ),
    _lang("uk", "Ukrainian", "Українська"),
    _lang("ur", "Urdu", "اردو"),
    _lang(
        "ug", "Uyghur", "ئۇيغۇرچە", "Uighur", "Uygur",
        descriptor="Uyghur, written in Arabic script",
    ),
    _lang(
        "uz", "Uzbek", "Oʻzbekcha",
        descriptor="Uzbek, written in Latin script",
    ),
    _lang("vi", "Vietnamese", "Tiếng Việt"),
    _lang("cy", "Welsh", "Cymraeg"),
    _lang("fy", "West Frisian", "Frysk", "Frisian", "Western Frisian"),
    _lang("wo", "Wolof"),
    _lang("xh", "Xhosa", "isiXhosa"),
    _lang("yi", "Yiddish", "ייִדיש"),
    _lang("yo", "Yoruba", "Yorùbá"),
    _lang("zu", "Zulu", "isiZulu"),
)

STANDARD_LANGUAGES: tuple[LanguageEntry, ...] = _PINNED_LANGUAGES + tuple(
    sorted(_OTHER_LANGUAGES, key=lambda entry: entry.name.casefold())
)

# Zero-width (non-)joiners are format characters that some scripts need inside
# words; every other control or format character is removed from names.
_KEPT_FORMAT_CHARACTERS = frozenset("‌‍")
_EDGE_QUOTES = "\"'`“”‘’„‚«»‹›"
# A normalized name with one trailing parenthetical qualifier, such as
# "persian (farsi)". NFKC normalization folds full-width parentheses.
_PARENTHETICAL_NAME_RE = re.compile(r"^(?P<outside>[^()]*?)\s*\((?P<inside>[^()]*)\)$")


def _without_control_characters(text: str) -> str:
    cleaned = []
    for character in text:
        if character.isspace():
            cleaned.append(" ")
            continue
        category = unicodedata.category(character)
        if category.startswith("C") and character not in _KEPT_FORMAT_CHARACTERS:
            continue
        cleaned.append(character)
    return "".join(cleaned)


def normalize_language_name(name: str) -> str:
    """Return the comparison form of a language name.

    Casefolds, removes control characters, collapses whitespace, and drops
    surrounding quotes and trailing punctuation such as "Spanish." while keeping
    a closing bracket, so "Chinese (Simplified)" stays recognizable.
    """
    text = unicodedata.normalize("NFKC", name if isinstance(name, str) else "")
    text = " ".join(_without_control_characters(text).casefold().split())
    while text and (
        text[-1] in _EDGE_QUOTES
        or (unicodedata.category(text[-1]).startswith("P") and unicodedata.category(text[-1]) != "Pe")
    ):
        text = text[:-1].rstrip()
    while text and text[0] in _EDGE_QUOTES:
        text = text[1:].lstrip()
    return text


def _compact_language_name(normalized: str) -> str:
    """Letters, marks, and digits only, so spacing and punctuation never matter."""
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] in ("L", "M", "N")
    )


def _build_name_indexes() -> tuple[dict[str, tuple[LanguageEntry, ...]], dict[str, tuple[LanguageEntry, ...]]]:
    exact: dict[str, list[LanguageEntry]] = {}
    compact: dict[str, list[LanguageEntry]] = {}
    for entry in STANDARD_LANGUAGES:
        for form in (entry.name, entry.native_name, *entry.aliases):
            normalized = normalize_language_name(form)
            if not normalized:
                continue
            for index, index_key in ((exact, normalized), (compact, _compact_language_name(normalized))):
                matches = index.setdefault(index_key, [])
                if entry not in matches:
                    matches.append(entry)
    return (
        {key: tuple(value) for key, value in exact.items()},
        {key: tuple(value) for key, value in compact.items() if key},
    )


_BY_KEY = {entry.key: entry for entry in STANDARD_LANGUAGES}
_BY_FOLDED_KEY = {entry.key.casefold(): entry for entry in STANDARD_LANGUAGES}
_BY_EXACT_NAME, _BY_COMPACT_NAME = _build_name_indexes()


def get_language(key: str) -> LanguageEntry | None:
    """Return the standard catalog entry for a key (case-insensitive), if any."""
    if not isinstance(key, str):
        return None
    cleaned = key.strip()
    return _BY_KEY.get(cleaned) or _BY_FOLDED_KEY.get(cleaned.casefold())


def _match_normalized(normalized: str) -> tuple[LanguageEntry, ...]:
    if not normalized:
        return ()
    exact = _BY_EXACT_NAME.get(normalized)
    if exact:
        return exact
    return _BY_COMPACT_NAME.get(_compact_language_name(normalized), ())


def match_language_names(name: str) -> tuple[LanguageEntry, ...]:
    """Return catalog entries whose name, endonym, or alias matches ``name``.

    An exact normalized match wins; otherwise spacing and punctuation are
    ignored. A name with a parenthetical qualifier that matches nothing as a
    whole, such as "Persian (Farsi)" or "Greek (Modern)", falls back to the
    text outside the parentheses and then the text inside them. Shared aliases
    deliberately match several entries ("Portuguese" matches both Portuguese
    variants).
    """
    normalized = normalize_language_name(name)
    matches = _match_normalized(normalized)
    if matches:
        return matches
    qualified = _PARENTHETICAL_NAME_RE.match(normalized)
    if qualified is None:
        return ()
    for part in (qualified.group("outside"), qualified.group("inside")):
        matches = _match_normalized(normalize_language_name(part))
        if matches:
            return matches
    return ()


def sanitize_custom_language_name(name: str) -> str:
    """Return a safe display name for a custom or model-reported language.

    Removes control characters, collapses whitespace, and caps the name at
    60 characters. Raises ValueError when no letter remains.
    """
    if not isinstance(name, str):
        raise ValueError("Custom language name must be text.")
    text = " ".join(_without_control_characters(unicodedata.normalize("NFC", name)).split())
    text = text[:CUSTOM_LANGUAGE_NAME_MAX_CHARS].rstrip()
    if not text:
        raise ValueError("Custom language name is empty.")
    if not any(unicodedata.category(character).startswith("L") for character in text):
        raise ValueError("Custom language name must contain at least one letter.")
    return text


def custom_language_key(name: str) -> str:
    """Return the deterministic storage key for a custom language name."""
    digest = hashlib.sha256(normalize_language_name(name).encode("utf-8")).hexdigest()
    return f"{CUSTOM_LANGUAGE_KEY_PREFIX}{digest[:CUSTOM_LANGUAGE_KEY_HASH_CHARS]}"


def resolve_language(key: str, name: str | None = None) -> LanguageEntry:
    """Resolve a settings language to its catalog entry or a custom entry.

    Standard keys return the catalog entry (the given name is ignored). Custom
    keys must equal the key recomputed from the sanitized name. Raises
    ValueError for unknown keys, invalid custom names, or mismatched keys.
    """
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Language key is required.")
    cleaned_key = key.strip()
    if cleaned_key.casefold().startswith(CUSTOM_LANGUAGE_KEY_PREFIX):
        if name is None:
            raise ValueError("Custom language name is required.")
        cleaned_name = sanitize_custom_language_name(name)
        expected_key = custom_language_key(cleaned_name)
        if cleaned_key != expected_key:
            raise ValueError("Custom language key does not match its name.")
        return LanguageEntry(key=expected_key, name=cleaned_name, native_name="")
    entry = get_language(cleaned_key)
    if entry is None:
        raise ValueError("Unknown language key.")
    return entry


def language_descriptor(entry: LanguageEntry) -> str:
    """Return the language description used in translation prompts."""
    return entry.descriptor or entry.name


def _classify_reported_names(names: Iterable[str] | None) -> tuple[set[str], dict[str, str]]:
    """Split one model's reported names into catalog keys and unmatched names.

    Unmatched names are keyed by their compact comparison form and keep the
    first sanitized spelling.
    """
    keys: set[str] = set()
    extras: dict[str, str] = {}
    for name in names or ():
        if not isinstance(name, str):
            continue
        try:
            cleaned = sanitize_custom_language_name(name)
        except ValueError:
            continue
        matches = match_language_names(name) or match_language_names(cleaned)
        if matches:
            keys.update(entry.key for entry in matches)
            continue
        compact = _compact_language_name(normalize_language_name(cleaned))
        if compact and compact not in extras:
            extras[compact] = cleaned
    return keys, extras


def proficiency_from_probe(
    validator_names: Iterable[str] | None,
    submitter_name_lists: Sequence[Iterable[str] | None] | None,
) -> tuple[set[str], list[str]]:
    """Combine probe answers into proficient catalog keys and extra languages.

    A catalog key is proficient when the validator model lists it and at least
    one submitter model lists it. Extras are languages outside the catalog that
    the validator and at least one submitter both list, returned with the
    validator's sanitized spelling, deduplicated and sorted case-insensitively.
    """
    validator_keys, validator_extras = _classify_reported_names(validator_names)
    submitter_keys: set[str] = set()
    submitter_extras: set[str] = set()
    for names in submitter_name_lists or ():
        keys, extras = _classify_reported_names(names)
        submitter_keys.update(keys)
        submitter_extras.update(extras)
    extras = sorted(
        (display for compact, display in validator_extras.items() if compact in submitter_extras),
        key=lambda display: (display.casefold(), display),
    )
    return validator_keys & submitter_keys, extras
