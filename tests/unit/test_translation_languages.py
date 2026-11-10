import re

import pytest

from backend.shared.translation.languages import (
    PINNED_LANGUAGE_KEYS,
    STANDARD_LANGUAGES,
    LanguageEntry,
    custom_language_key,
    get_language,
    language_descriptor,
    match_language_names,
    normalize_language_name,
    proficiency_from_probe,
    resolve_language,
    sanitize_custom_language_name,
)


REQUIRED_CATALOG_NAMES = (
    "Afrikaans", "Albanian", "Amharic", "Armenian", "Assamese", "Azerbaijani", "Basque",
    "Belarusian", "Bengali", "Bosnian", "Bulgarian", "Burmese", "Cantonese", "Catalan",
    "Cebuano", "Chinese (Traditional)", "Croatian", "Czech", "Danish", "Dutch", "Esperanto",
    "Estonian", "Filipino", "Finnish", "Galician", "Georgian", "German", "Greek", "Gujarati",
    "Haitian Creole", "Hausa", "Hebrew", "Hungarian", "Icelandic", "Igbo", "Indonesian",
    "Irish", "Italian", "Japanese", "Javanese", "Kannada", "Kazakh", "Khmer", "Kinyarwanda",
    "Korean", "Kurdish (Kurmanji)", "Kyrgyz", "Lao", "Latin", "Latvian", "Lithuanian",
    "Luxembourgish", "Macedonian", "Malagasy", "Malay", "Malayalam", "Maltese", "Maori",
    "Marathi", "Mongolian", "Nepali", "Norwegian (Bokmål)", "Odia", "Pashto", "Persian",
    "Polish", "Portuguese (Brazil)", "Portuguese (Portugal)", "Punjabi", "Quechua",
    "Romanian", "Russian", "Samoan", "Scottish Gaelic", "Serbian", "Shona", "Sindhi",
    "Sinhala", "Slovak", "Slovenian", "Somali", "Swahili", "Swedish", "Tajik", "Tamil",
    "Tatar", "Telugu", "Thai", "Tibetan", "Tigrinya", "Turkish", "Turkmen", "Ukrainian",
    "Urdu", "Uyghur", "Uzbek", "Vietnamese", "Welsh", "Xhosa", "Yiddish", "Yoruba", "Zulu",
)


def _keys(entries):
    return [entry.key for entry in entries]


def test_pinned_languages_come_first_in_exact_order_and_names():
    pinned = STANDARD_LANGUAGES[: len(PINNED_LANGUAGE_KEYS)]

    assert PINNED_LANGUAGE_KEYS == ("en", "zh-Hans", "hi", "es", "ar", "fr")
    assert _keys(pinned) == list(PINNED_LANGUAGE_KEYS)
    assert [entry.name for entry in pinned] == [
        "English",
        "Mandarin Chinese",
        "Hindi",
        "Spanish",
        "Modern Standard Arabic",
        "French",
    ]
    assert all(entry.pinned for entry in pinned)
    assert not any(entry.pinned for entry in STANDARD_LANGUAGES[len(PINNED_LANGUAGE_KEYS):])
    assert "Simplified Chinese characters" in get_language("zh-Hans").descriptor


def test_unpinned_languages_are_sorted_by_english_name():
    rest = [entry.name for entry in STANDARD_LANGUAGES[len(PINNED_LANGUAGE_KEYS):]]

    assert rest == sorted(rest, key=str.casefold)
    assert len(STANDARD_LANGUAGES) >= 100


def test_catalog_keys_and_names_are_unique_and_cover_required_languages():
    keys = [entry.key for entry in STANDARD_LANGUAGES]
    folded_names = [entry.name.casefold() for entry in STANDARD_LANGUAGES]
    names = {entry.name for entry in STANDARD_LANGUAGES}

    assert len(keys) == len(set(keys))
    assert len({key.casefold() for key in keys}) == len(keys)
    assert len(folded_names) == len(set(folded_names))
    assert set(REQUIRED_CATALOG_NAMES) <= names
    assert get_language("zh-yue").name == "Cantonese"
    assert get_language("zh-Hant").name == "Chinese (Traditional)"
    assert {get_language("pt-BR").name, get_language("pt-PT").name} == {
        "Portuguese (Brazil)",
        "Portuguese (Portugal)",
    }


def test_catalog_entries_are_well_formed():
    for entry in STANDARD_LANGUAGES:
        assert entry.key and entry.key == entry.key.strip()
        assert not entry.key.startswith("custom-")
        assert entry.name == entry.name.strip()
        assert len(entry.name) <= 60
        assert entry.native_name != entry.name
        assert entry.name not in entry.aliases
        assert len(set(entry.aliases)) == len(entry.aliases)
        assert isinstance(entry.aliases, tuple)


def test_each_english_name_and_endonym_matches_only_its_own_entry():
    for entry in STANDARD_LANGUAGES:
        assert match_language_names(entry.name) == (entry,)
        if entry.native_name:
            assert match_language_names(entry.native_name) == (entry,)


def test_only_portuguese_aliases_are_shared_between_entries():
    shared = {}
    for entry in STANDARD_LANGUAGES:
        for form in (entry.name, entry.native_name, *entry.aliases):
            if form:
                shared.setdefault(normalize_language_name(form), set()).add(entry.key)
    ambiguous = {form: keys for form, keys in shared.items() if len(keys) > 1}

    assert ambiguous == {
        "portuguese": {"pt-BR", "pt-PT"},
        "português": {"pt-BR", "pt-PT"},
    }


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("Portuguese", ["pt-BR", "pt-PT"]),
        ("Chinese", ["zh-Hans"]),
        ("Chinese (Simplified)", ["zh-Hans"]),
        ("Simplified Chinese", ["zh-Hans"]),
        ("Mandarin", ["zh-Hans"]),
        ("Traditional Chinese", ["zh-Hant"]),
        ("Arabic", ["ar"]),
        ("Standard Arabic", ["ar"]),
        ("Norwegian", ["nb"]),
        ("Tagalog", ["fil"]),
        ("Farsi", ["fa"]),
        ("Kurdish", ["ku"]),
        ("Malay", ["ms"]),
        ("Malayalam", ["ml"]),
        ("Brazilian Portuguese", ["pt-BR"]),
        ("Español", ["es"]),
        ("日本語", ["ja"]),
    ],
)
def test_reported_names_match_catalog_entries(reported, expected):
    assert _keys(match_language_names(reported)) == expected


@pytest.mark.parametrize(
    "reported",
    [
        "chinese (simplified)",
        "  CHINESE   (Simplified)  ",
        "Chinese (Simplified).",
        "Chinese Simplified",
        "chinese-simplified",
        "\"Chinese, Simplified\"",
        "Chinese\t(Simplified)\n",
    ],
)
def test_matching_ignores_case_whitespace_and_punctuation(reported):
    assert _keys(match_language_names(reported)) == ["zh-Hans"]


@pytest.mark.parametrize("reported", ["", "   ", "...", "Klingon", "Elvish (Sindarin)", None, 42])
def test_unknown_or_empty_names_match_nothing(reported):
    assert match_language_names(reported) == ()


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("Persian (Farsi)", ["fa"]),
        ("Arabic (Modern Standard)", ["ar"]),
        ("Hindi (Devanagari)", ["hi"]),
        ("Greek (Modern)", ["el"]),
        ("Filipino (Tagalog)", ["fil"]),
        ("Bengali (Bangla)", ["bn"]),
        ("Portuguese (Brazilian)", ["pt-BR"]),
        ("Portuguese (European)", ["pt-PT"]),
        ("European Portuguese", ["pt-PT"]),
        # The text outside the parentheses is tried before the qualifier.
        ("Uzbek (Latin)", ["uz"]),
        # Only the qualifier names a catalog language.
        ("Pilipino (Tagalog)", ["fil"]),
        # Whole-name matches still win over the fallback.
        ("Kurdish (Sorani)", ["ckb"]),
        ("Chinese (Traditional)", ["zh-Hant"]),
        ("Ｐｅｒｓｉａｎ（Ｆａｒｓｉ）", ["fa"]),
    ],
)
def test_parenthetical_qualifiers_fall_back_to_outside_then_inside_text(reported, expected):
    assert _keys(match_language_names(reported)) == expected


def test_probe_answers_with_parenthetical_qualifiers_mark_standard_languages_proficient():
    names = [
        "Persian (Farsi)",
        "Arabic (Modern Standard)",
        "Hindi (Devanagari)",
        "Greek (Modern)",
        "Portuguese (Brazilian)",
    ]
    proficient, extras = proficiency_from_probe(names, [names])
    assert proficient == {"fa", "ar", "hi", "el", "pt-BR"}
    assert extras == []


def test_normalize_language_name_rules():
    assert normalize_language_name("  Modern   STANDARD Arabic. ") == "modern standard arabic"
    assert normalize_language_name("Chinese (Simplified)!") == "chinese (simplified)"
    assert normalize_language_name("“French”") == "french"
    assert normalize_language_name("Old\x00 Norse") == "old norse"
    assert normalize_language_name(None) == ""


def test_get_language_lookup():
    assert get_language("zh-Hans").name == "Mandarin Chinese"
    assert get_language("ZH-hans") is get_language("zh-Hans")
    assert get_language(" fr ") is get_language("fr")
    assert get_language("xx") is None
    assert get_language("") is None
    assert get_language(None) is None


def test_sanitize_custom_language_name_strips_controls_and_collapses_whitespace():
    assert sanitize_custom_language_name("  Old\x00 \x07Norse\x1b ") == "Old Norse"
    assert sanitize_custom_language_name("Old\nNorse\tRunic") == "Old Norse Runic"
    assert sanitize_custom_language_name("Lat‮IN") == "LatIN"
    assert sanitize_custom_language_name("क्‍ष") == "क्‍ष"


def test_sanitize_custom_language_name_caps_length():
    long_name = "Proto-Indo-European " * 10

    cleaned = sanitize_custom_language_name(long_name)

    assert len(cleaned) <= 60
    assert cleaned == long_name[:60].rstrip()
    assert cleaned == cleaned.strip()


@pytest.mark.parametrize("name", ["", "   ", "\x00\x01", "12345", "--- !!!", "1" * 70 + "Norse", None, 7])
def test_sanitize_custom_language_name_rejects_names_without_letters(name):
    with pytest.raises(ValueError):
        sanitize_custom_language_name(name)


def test_custom_language_key_is_deterministic_and_normalized():
    key = custom_language_key("Old Norse")

    assert re.fullmatch(r"custom-[0-9a-f]{10}", key)
    assert custom_language_key("Old Norse") == key
    assert custom_language_key("  old   NORSE. ") == key
    assert custom_language_key("Old\x00 Norse") == key
    assert custom_language_key("Old English") != key


def test_resolve_language_returns_catalog_entries_for_standard_keys():
    entry = resolve_language("es", "Castellano personalizado")

    assert entry is get_language("es")
    assert entry.name == "Spanish"
    assert resolve_language("ZH-HANS", None) is get_language("zh-Hans")


def test_resolve_language_builds_custom_entries_from_sanitized_names():
    key = custom_language_key("Old Norse")

    entry = resolve_language(key, "  Old\x00   Norse ")

    assert entry == LanguageEntry(key=key, name="Old Norse", native_name="")
    assert not entry.pinned
    assert language_descriptor(entry) == "Old Norse"


@pytest.mark.parametrize(
    ("key", "name"),
    [
        (custom_language_key("Old Norse"), "Old English"),
        (custom_language_key("Old Norse").upper(), "Old Norse"),
        ("custom-0000000000", "Old Norse"),
        ("custom-", "Old Norse"),
        (custom_language_key("Old Norse"), None),
        (custom_language_key("12345"), "12345"),
        ("xx-unknown", None),
        ("", "English"),
        ("   ", None),
        (None, "English"),
    ],
)
def test_resolve_language_rejects_invalid_keys(key, name):
    with pytest.raises(ValueError):
        resolve_language(key, name)


def test_language_descriptor_prefers_descriptor_then_name():
    assert language_descriptor(get_language("zh-Hans")) == (
        "Mandarin Chinese, written in Simplified Chinese characters"
    )
    assert language_descriptor(get_language("en")) == "English"
    assert language_descriptor(get_language("ar")) == "Modern Standard Arabic"


def test_probe_requires_validator_and_at_least_one_submitter():
    proficient, extras = proficiency_from_probe(
        ["Spanish", "French", "German", "Japanese"],
        [["spanish", "Japanese"], ["Français"], ["Korean"]],
    )

    assert proficient == {"es", "fr", "ja"}
    assert extras == []


def test_probe_ignores_languages_missing_from_the_validator():
    proficient, _ = proficiency_from_probe(["English"], [["English", "Korean"], ["Korean"]])

    assert proficient == {"en"}


def test_probe_without_submitters_marks_nothing_proficient():
    assert proficiency_from_probe(["English", "Klingon"], []) == (set(), [])
    assert proficiency_from_probe(["English"], None) == (set(), [])
    assert proficiency_from_probe(None, [["English"]]) == (set(), [])


def test_probe_matches_aliases_and_intersects_shared_aliases():
    proficient, _ = proficiency_from_probe(
        ["Chinese", "Portuguese", "Arabic"],
        [["Mandarin Chinese", "Brazilian Portuguese"], ["MSA"]],
    )

    assert proficient == {"zh-Hans", "pt-BR", "ar"}


def test_probe_extras_are_shared_unmatched_names_deduped_sorted_and_sanitized():
    proficient, extras = proficiency_from_probe(
        [
            "Klingon",
            "klingon.",
            "old norse",
            "Ancient\x00 Greek",
            "Elvish",
            "Spanish",
            "Sanskrit" + " " * 3 + "x" * 80,
            "12345",
            None,
            7,
        ],
        [
            ["KLINGON"],
            ["Old-Norse", "Ancient Greek", "Spanish", "Sanskrit " + "x" * 80],
            ["Dothraki"],
        ],
    )

    assert proficient == {"es"}
    assert extras == sorted(extras, key=str.casefold)
    assert extras == ["Ancient Greek", "Klingon", "old norse", ("Sanskrit " + "x" * 80)[:60]]
    assert all(len(name) <= 60 for name in extras)


def test_probe_does_not_report_catalog_languages_as_extras():
    proficient, extras = proficiency_from_probe(["Spanish\x00", "Tagalog"], [["spanish"], ["Filipino"]])

    assert proficient == {"es", "fil"}
    assert extras == []
