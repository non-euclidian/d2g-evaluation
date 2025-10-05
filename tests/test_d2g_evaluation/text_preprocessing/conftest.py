import pytest

from d2g_evaluation.text_preprocessing.text_preprocessing_core import (
    ImplementedStringPreprocessing,
    ImplementedTokenization,
)

TEST_DECODE_UNICODE_ESCAPES = [
    pytest.param(
        "hello\nworld",
        "hello\nworld",
        id="newline",
    ),
    pytest.param(
        "hello\tworld",
        "hello\tworld",
        id="tab",
    ),
    pytest.param(
        "hello\\nworld",
        "hello\nworld",
        id="escaped-backslash-n",
    ),
    pytest.param(
        "emoji:\u2764",
        "emoji:❤",
        id="unicode-heart",
    ),
    pytest.param(
        "emoji:\U0001f600",
        "emoji:😀",
        id="unicode-emoji",
    ),
    pytest.param(
        "\x48\x69",
        "Hi",
        id="hex-H-i",
    ),
    pytest.param(
        "\101\102\103",
        "ABC",
        id="octal-ABC",
    ),
    pytest.param(
        "line1\\nline2\nline3",
        """line1\nline2\nline3""",
        id="mixed-newlines",
    ),
    pytest.param(
        "slash:\\\\",
        "slash:\\",
        id="escaped-backslash",
    ),
    # multilingual
    pytest.param(
        "привет\nмир",  # noqa: RUF001
        "привет\nмир",  # noqa: RUF001
        id="cyrillic",
    ),
    pytest.param(
        "こんにちは\n世界",
        "こんにちは\n世界",
        id="japanese",
    ),
    pytest.param(
        "你好\n世界",
        "你好\n世界",
        id="chinese",
    ),
    pytest.param(
        "שָׁלוֹם\nעוֹלָם",  # noqa: RUF001
        "שָׁלוֹם\nעוֹלָם",  # noqa: RUF001
        id="hebrew",
    ),
    pytest.param(
        "مرحبا\nبالعالم",  # noqa: RUF001
        "مرحبا\nبالعالم",  # noqa: RUF001
        id="arabic",
    ),
    pytest.param(
        "hello\\nworld",
        "hello\nworld",
        id="newline",
    ),
    pytest.param(
        "hello\\tworld",
        "hello\tworld",
        id="tab",
    ),
    pytest.param(
        "hello\\rworld",
        "hello\rworld",
        id="carriage",
    ),
    pytest.param(
        "hello\\x41world",
        "helloAworld",
        id="hex",
    ),
    pytest.param(
        "hello\\u0041world",
        "helloAworld",
        id="unicode",
    ),
    pytest.param(
        "hello\\U00000041world",
        "helloAworld",
        id="unicode_full",
    ),
    pytest.param(
        "hello\\U0001F600world",
        "hello😀world",
        id="emoji",
    ),
    pytest.param(
        "",
        "",
        id="empty_string",
    ),
    pytest.param(
        "hello\\nworld\\t",
        "hello\nworld\t",
        id="newline+tab",
    ),
    pytest.param(
        "hello\\nworld\\r",
        "hello\nworld\r",
        id="newline+carriage",
    ),
    pytest.param(
        "hello\\nworld\\x41",
        "hello\nworldA",
        id="newline+hex",
    ),
    pytest.param(
        "hello\\nworld\\u0041",
        "hello\nworldA",
        id="newline+unicode",
    ),
    pytest.param(
        "hello\\nworld\\U00000041",
        "hello\nworldA",
        id="newline+unicode_full",
    ),
    pytest.param(
        "hello\\nworld\\U0001F600",
        "hello\nworld😀",
        id="newline+emoji",
    ),
]

TEST_NORMALIZE_WHITESPACES = [
    pytest.param(
        "   hello   world   ",
        "hello world",
        " ",
        True,
        id="space+strip_True",
    ),
    pytest.param(
        "   hello   world   ",
        " hello world ",
        " ",
        False,
        id="space+strip_False",
    ),
    pytest.param(
        "   hello   world   ",
        "helloworld",
        "",
        True,
        id="empty_string+strip_True",
    ),
    pytest.param(
        "   hello   world   ",
        "helloworld",
        "",
        False,
        id="empty_string+strip_False",
    ),
    pytest.param(
        "",
        "",
        " ",
        True,
        id="input_empty_string_+space+strip_True",
    ),
    pytest.param(
        "",
        "",
        " ",
        False,
        id="input_empty_string_+space+strip_False",
    ),
    pytest.param(
        " ",
        "",
        " ",
        True,
        id="input_space_+space+strip_True",
    ),
    pytest.param(
        " ",
        " ",
        " ",
        False,
        id="input_space_+space+strip_False",
    ),
    pytest.param(
        " ",
        "",
        " ",
        True,
        id="input_space_+space+strip_True",
    ),
    pytest.param(
        " ",
        " ",
        " ",
        False,
        id="input_space_+space+strip_False",
    ),
    pytest.param(
        "   hello   world   ",
        "-hello-world-",
        "-",
        True,
        id="dash+strip_True",
    ),
    pytest.param(
        "   hello   world   ",
        "-hello-world-",
        "-",
        False,
        id="dash+strip_False",
    ),
]


TEST_NORMALIZE_STRING = [
    pytest.param("hello   world", "hello world", " ", True, id="space+strip_True"),
    pytest.param("  hello   world  ", " hello world ", " ", False, id="space+strip_False"),
    pytest.param("  hello   world  ", "helloworld", "", True, id="empty_string+strip_True"),
    pytest.param("  hello   world  ", "helloworld", "", False, id="empty_string+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hi world", " ", True, id="space+strip_True"),
    pytest.param("   hi\\n\\tworld   ", " hi world ", " ", False, id="space+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hiworld", "", True, id="empty_string+strip_True"),
    pytest.param("  hi\\n\\tworld ", "hiworld", "", False, id="empty_string+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hi world", " ", True, id="space+strip_True"),
    pytest.param("   hi\\n\\tworld   ", " hi world ", " ", False, id="space+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hiworld", "", True, id="empty_string+strip_True"),
    pytest.param("  hi\\n\\tworld ", "hiworld", "", False, id="empty_string+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hi world", " ", True, id="space+strip_True"),
    pytest.param("   hi\\n\\tworld   ", " hi world ", " ", False, id="space+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hiworld", "", True, id="empty_string+strip_True"),
    pytest.param("  hi\\n\\tworld ", "hiworld", "", False, id="empty_string+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hi world", " ", True, id="space+strip_True"),
    pytest.param("   hi\\n\\tworld   ", " hi world ", " ", False, id="space+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hiworld", "", True, id="empty_string+strip_True"),
]


TEST_NORMALIZE_STRING = [
    pytest.param("hello   world", "hello world", " ", True, id="space+strip_True"),
    pytest.param("  hello   world  ", " hello world ", " ", False, id="space+strip_False"),
    pytest.param("  hello   world  ", "helloworld", "", True, id="empty_string+strip_True"),
    pytest.param("  hello   world  ", "helloworld", "", False, id="empty_string+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hi world", " ", True, id="space+strip_True"),
    pytest.param("   hi\\n\\tworld   ", " hi world ", " ", False, id="space+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hiworld", "", True, id="empty_string+strip_True"),
    pytest.param("  hi\\n\\tworld ", "hiworld", "", False, id="empty_string+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hi world", " ", True, id="space+strip_True"),
    pytest.param("   hi\\n\\tworld   ", " hi world ", " ", False, id="space+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hiworld", "", True, id="empty_string+strip_True"),
    pytest.param("  hi\\n\\tworld ", "hiworld", "", False, id="empty_string+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hi world", " ", True, id="space+strip_True"),
    pytest.param("   hi\\n\\tworld   ", " hi world ", " ", False, id="space+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hiworld", "", True, id="empty_string+strip_True"),
    pytest.param("  hi\\n\\tworld ", "hiworld", "", False, id="empty_string+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hi world", " ", True, id="space+strip_True"),
    pytest.param("   hi\\n\\tworld   ", " hi world ", " ", False, id="space+strip_False"),
    pytest.param(" hi\\n\\tworld ", "hiworld", "", True, id="empty_string+strip_True"),
]


TEST_TOKENIZE_BY_CHAR_NGRAMS = [
    pytest.param(
        "hello my name",
        1,
        ["h", "e", "l", "l", "o", " ", "m", "y", " ", "n", "a", "m", "e"],
        id="1_char_ngrams",
    ),
    pytest.param(
        "hello my name",
        2,
        ["he", "el", "ll", "lo", "o ", " m", "my", "y ", " n", "na", "am", "me"],
        id="2_char_ngrams",
    ),
    pytest.param(
        "hello my name",
        3,
        ["hel", "ell", "llo", "lo ", "o m", " my", "my ", "y n", " na", "nam", "ame"],
        id="3_char_ngrams",
    ),
    pytest.param(
        "hello my name",
        4,
        ["hell", "ello", "llo ", "lo m", "o my", " my ", "my n", "y na", " nam", "name"],
        id="4_char_ngrams",
    ),
    pytest.param(
        "hello my name",
        5,
        ["hello", "ello ", "llo m", "lo my", "o my ", " my n", "my na", "y nam", " name"],
        id="5_char_ngrams",
    ),
    pytest.param(
        "",
        1,
        [],
        id="empty_string_1_char_ngrams",
    ),
    pytest.param(
        "",
        2,
        [],
        id="empty_string_2_char_ngrams",
    ),
    pytest.param(
        "",
        3,
        [],
        id="empty_string_3_char_ngrams",
    ),
    pytest.param(
        "",
        4,
        [],
        id="empty_string_4_char_ngrams",
    ),
    pytest.param(
        "",
        5,
        [],
        id="empty_string_5_char_ngrams",
    ),
    pytest.param(
        "",
        1000,
        [],
        id="empty_string_1000_char_ngrams",
    ),
    pytest.param(
        "wine",
        1,
        ["w", "i", "n", "e"],
        id="1_char_ngrams_wine",
    ),
    pytest.param(
        "wine",
        2,
        ["wi", "in", "ne"],
        id="2_char_ngrams_wine",
    ),
    pytest.param(
        "wine",
        3,
        ["win", "ine"],
        id="3_char_ngrams_wine",
    ),
    pytest.param(
        "wine",
        4,
        ["wine"],
        id="4_char_ngrams_wine",
    ),
    pytest.param(
        "wine",
        5,
        ["wine"],
        id="5_char_ngrams_wine",
    ),
    pytest.param(
        "wine",
        1000,
        ["wine"],
        id="1000_char_ngrams_wine",
    ),
    pytest.param(
        "     ",
        1,
        [" ", " ", " ", " ", " "],
        id="1_char_ngrams_space",
    ),
    pytest.param(
        "     ",
        2,
        ["  ", "  ", "  ", "  "],
        id="2_char_ngrams_space",
    ),
    pytest.param(
        "     ",
        3,
        ["   ", "   ", "   "],
        id="3_char_ngrams_space",
    ),
    pytest.param(
        "     ",
        4,
        ["    ", "    "],
        id="4_char_ngrams_space",
    ),
    pytest.param(
        "     ",
        5,
        ["     "],
        id="5_char_ngrams_space",
    ),
    pytest.param(
        "     ",
        6,
        ["     "],
        id="6_char_ngrams_space",
    ),
    pytest.param(
        "     ",
        1000,
        ["     "],
        id="1000_char_ngrams_space",
    ),
]


TEST_TOKENIZE_BY_NCHARS = [
    pytest.param(
        "hello my name",
        1,
        ["h", "e", "l", "l", "o", " ", "m", "y", " ", "n", "a", "m", "e"],
        id="1_nchars",
    ),
    pytest.param(
        "hello my name",
        2,
        ["he", "ll", "o ", "my", " n", "am", "e"],
        id="2_nchars",
    ),
    pytest.param(
        "hello my name",
        3,
        ["hel", "lo ", "my ", "nam", "e"],
        id="3_nchars",
    ),
    pytest.param(
        "hello my name",
        4,
        ["hell", "o my", " nam", "e"],
        id="4_nchars",
    ),
    pytest.param(
        "hello my name",
        5,
        ["hello", " my n", "ame"],
        id="5_nchars",
    ),
    pytest.param(
        "hello my name",
        1000,
        ["hello my name"],
        id="1000_nchars",
    ),
    pytest.param(
        "",
        1,
        [],
        id="empty_string_1_nchars",
    ),
    pytest.param(
        "",
        2,
        [],
        id="empty_string_2_nchars",
    ),
    pytest.param(
        "",
        3,
        [],
        id="empty_string_3_nchars",
    ),
    pytest.param(
        "",
        4,
        [],
        id="empty_string_4_nchars",
    ),
    pytest.param(
        "",
        5,
        [],
        id="empty_string_5_nchars",
    ),
    pytest.param(
        "",
        1000,
        [],
        id="empty_string_1000_nchars",
    ),
    pytest.param(
        "wine",
        1,
        ["w", "i", "n", "e"],
        id="1_nchars_wine",
    ),
    pytest.param(
        "wine",
        2,
        ["wi", "ne"],
        id="2_nchars_wine",
    ),
    pytest.param(
        "wine",
        3,
        ["win", "e"],
        id="3_nchars_wine",
    ),
    pytest.param(
        "wine",
        4,
        ["wine"],
        id="4_nchars_wine",
    ),
    pytest.param(
        "wine",
        5,
        ["wine"],
        id="5_nchars_wine",
    ),
    pytest.param(
        "wine",
        1000,
        ["wine"],
        id="1000_nchars_wine",
    ),
    pytest.param(
        "     ",
        1,
        [" ", " ", " ", " ", " "],
        id="1_nchars_space",
    ),
    pytest.param(
        "     ",
        2,
        ["  ", "  ", " "],
        id="2_nchars_space",
    ),
    pytest.param(
        "     ",
        3,
        ["   ", "  "],
        id="3_nchars_space",
    ),
    pytest.param(
        "     ",
        4,
        ["    ", " "],
        id="4_nchars_space",
    ),
    pytest.param(
        "     ",
        5,
        ["     "],
        id="5_nchars_space",
    ),
    pytest.param(
        "     ",
        6,
        ["     "],
        id="6_nchars_space",
    ),
    pytest.param(
        "     ",
        1000,
        ["     "],
        id="1000_nchars_space",
    ),
]


TEST_PREPROCESS_STRING = [
    pytest.param(
        "nice   week",
        "nice week",
        ImplementedStringPreprocessing.NORMALIZE_STRING,
        id="normalize",
    ),
    pytest.param(
        "   nice   week   ",
        "nice week",
        ImplementedStringPreprocessing.NORMALIZE_STRING,
        id="normalize",
    ),
    pytest.param(
        "123 lol 456",
        "123lol456",
        ImplementedStringPreprocessing.REMOVE_WHITESPACES,
        id="whitespace_removed",
    ),
    pytest.param(
        "   123 lol 456   ",
        "123lol456",
        ImplementedStringPreprocessing.REMOVE_WHITESPACES,
        id="whitespace_removed",
    ),
    pytest.param(
        "hello   world",
        "hello world",
        ImplementedStringPreprocessing.NORMALIZE_STRING,
        id="normalize",
    ),
    pytest.param(
        "hello\\nworld",
        "hello\nworld",
        ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
        id="unicode_escape",
    ),
    pytest.param(
        "hello\\tworld",
        "hello\tworld",
        ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
        id="unicode_escape",
    ),
    pytest.param(
        "hello\\rworld",
        "hello\rworld",
        ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
        id="unicode_escape",
    ),
    pytest.param(
        "hello   world",
        "hello world",
        ImplementedStringPreprocessing.NORMALIZE_STRING,
        id="replace_multiple_whitespaces",
    ),
    pytest.param(
        "  hello   world    ",
        "hello world",
        ImplementedStringPreprocessing.NORMALIZE_STRING,
        id="replace_multiple_whitespaces",
    ),
    pytest.param(
        "hello   world",
        "helloworld",
        ImplementedStringPreprocessing.REMOVE_WHITESPACES,
        id="whitespace_removed",
    ),
    pytest.param(
        "   hello   world  ",
        "helloworld",
        ImplementedStringPreprocessing.REMOVE_WHITESPACES,
        id="whitespace_removed",
    ),
    pytest.param(
        "нет привет\nмир",  # noqa: RUF001
        "нет привет\nмир",  # noqa: RUF001
        ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
        id="unicode_escape",
    ),
    pytest.param(
        "hello\\nworld",
        "hello\nworld",
        ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
        id="unicode_escape",
    ),
]


TEST_TOKENIZE_STRING = [
    pytest.param(
        "hello   world",
        ImplementedTokenization.CHAR_NGRAMS,
        3,
        ["hel", "ell", "llo", "lo ", "o  ", "   ", "  w", " wo", "wor", "orl", "rld"],
        id="char_ngrams",
    ),
    pytest.param("hello   world", ImplementedTokenization.NCHARS, 3, ["hel", "lo ", "  w", "orl", "d"], id="nchars"),
    pytest.param(
        "hello my name",
        ImplementedTokenization.CHAR_NGRAMS,
        2,
        ["he", "el", "ll", "lo", "o ", " m", "my", "y ", " n", "na", "am", "me"],
    ),
    pytest.param("hello my name", ImplementedTokenization.NCHARS, 3, ["hel", "lo ", "my ", "nam", "e"]),
]
