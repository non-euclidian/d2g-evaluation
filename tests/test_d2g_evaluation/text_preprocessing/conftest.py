from dataclasses import dataclass

from d2g_evaluation.text_preprocessing.text_preprocessing_core import (
    ImplementedStringPreprocessing,
    ImplementedTokenization,
)


@dataclass(frozen=True, slots=True)
class DecodeUnicodeTestCase:
    test_name: str
    input_text: str
    expected_text: str


TEST_DECODE_UNICODE_ESCAPES: list[DecodeUnicodeTestCase] = [
    DecodeUnicodeTestCase(test_name="newline", input_text="hello\nworld", expected_text="hello\nworld"),
    DecodeUnicodeTestCase(test_name="tab", input_text="hello\tworld", expected_text="hello\tworld"),
    DecodeUnicodeTestCase(test_name="escaped-backslash-n", input_text="hello\\nworld", expected_text="hello\nworld"),
    DecodeUnicodeTestCase(test_name="unicode-heart", input_text="emoji:\u2764", expected_text="emoji:❤"),
    DecodeUnicodeTestCase(test_name="unicode-emoji", input_text="emoji:\U0001f600", expected_text="emoji:😀"),
    DecodeUnicodeTestCase(test_name="hex-H-i", input_text="\x48\x69", expected_text="Hi"),
    DecodeUnicodeTestCase(test_name="octal-ABC", input_text="\101\102\103", expected_text="ABC"),
    DecodeUnicodeTestCase(
        test_name="mixed-newlines", input_text="line1\\nline2\nline3", expected_text="""line1\nline2\nline3"""
    ),
    DecodeUnicodeTestCase(test_name="escaped-backslash", input_text="slash:\\\\", expected_text="slash:\\"),
    # multilingual
    DecodeUnicodeTestCase(test_name="cyrillic", input_text="привет\nмир", expected_text="привет\nмир"),  # noqa: RUF001
    DecodeUnicodeTestCase(test_name="japanese", input_text="こんにちは\n世界", expected_text="こんにちは\n世界"),
    DecodeUnicodeTestCase(test_name="chinese", input_text="你好\n世界", expected_text="你好\n世界"),
    DecodeUnicodeTestCase(test_name="hebrew", input_text="שָׁלוֹם\nעוֹלָם", expected_text="שָׁלוֹם\nעוֹלָם"),  # noqa: RUF001
    DecodeUnicodeTestCase(test_name="arabic", input_text="مرحبا\nبالعالم", expected_text="مرحبا\nبالعالم"),  # noqa: RUF001
    DecodeUnicodeTestCase(test_name="newline", input_text="hello\\nworld", expected_text="hello\nworld"),
    DecodeUnicodeTestCase(test_name="tab", input_text="hello\\tworld", expected_text="hello\tworld"),
    DecodeUnicodeTestCase(test_name="carriage", input_text="hello\\rworld", expected_text="hello\rworld"),
    DecodeUnicodeTestCase(test_name="hex", input_text="hello\\x41world", expected_text="helloAworld"),
    DecodeUnicodeTestCase(test_name="unicode", input_text="hello\\u0041world", expected_text="helloAworld"),
    DecodeUnicodeTestCase(test_name="unicode_full", input_text="hello\\U00000041world", expected_text="helloAworld"),
    DecodeUnicodeTestCase(test_name="emoji", input_text="hello\\U0001F600world", expected_text="hello😀world"),
    DecodeUnicodeTestCase(test_name="empty_string", input_text="", expected_text=""),
    DecodeUnicodeTestCase(test_name="newline+tab", input_text="hello\\nworld\\t", expected_text="hello\nworld\t"),
    DecodeUnicodeTestCase(test_name="newline+carriage", input_text="hello\\nworld\\r", expected_text="hello\nworld\r"),
    DecodeUnicodeTestCase(test_name="newline+hex", input_text="hello\\nworld\\x41", expected_text="hello\nworldA"),
    DecodeUnicodeTestCase(
        test_name="newline+unicode", input_text="hello\\nworld\\u0041", expected_text="hello\nworldA"
    ),
    DecodeUnicodeTestCase(
        test_name="newline+unicode_full", input_text="hello\\nworld\\U00000041", expected_text="hello\nworldA"
    ),
    DecodeUnicodeTestCase(
        test_name="newline+emoji", input_text="hello\\nworld\\U0001F600", expected_text="hello\nworld😀"
    ),
]


@dataclass(frozen=True, slots=True)
class NormalizeWhitespacesTestCase:
    test_name: str
    input_text: str
    expected_text: str
    replacement: str
    strip: bool


TEST_NORMALIZE_WHITESPACES: list[NormalizeWhitespacesTestCase] = [
    NormalizeWhitespacesTestCase(
        test_name="space+strip_True",
        input_text="   hello   world   ",
        expected_text="hello world",
        replacement=" ",
        strip=True,
    ),
    NormalizeWhitespacesTestCase(
        test_name="space+strip_False",
        input_text="   hello   world   ",
        expected_text=" hello world ",
        replacement=" ",
        strip=False,
    ),
    NormalizeWhitespacesTestCase(
        test_name="empty_string+strip_True",
        input_text="   hello   world   ",
        expected_text="helloworld",
        replacement="",
        strip=True,
    ),
    NormalizeWhitespacesTestCase(
        test_name="empty_string+strip_False",
        input_text="   hello   world   ",
        expected_text="helloworld",
        replacement="",
        strip=False,
    ),
    NormalizeWhitespacesTestCase(
        test_name="input_empty_string_+space+strip_True", input_text="", expected_text="", replacement=" ", strip=True
    ),
    NormalizeWhitespacesTestCase(
        test_name="input_empty_string_+space+strip_False", input_text="", expected_text="", replacement=" ", strip=False
    ),
    NormalizeWhitespacesTestCase(
        test_name="input_space_+space+strip_True", input_text=" ", expected_text="", replacement=" ", strip=True
    ),
    NormalizeWhitespacesTestCase(
        test_name="input_space_+space+strip_False", input_text=" ", expected_text=" ", replacement=" ", strip=False
    ),
    NormalizeWhitespacesTestCase(
        test_name="dash+strip_True",
        input_text="   hello   world   ",
        expected_text="-hello-world-",
        replacement="-",
        strip=True,
    ),
    NormalizeWhitespacesTestCase(
        test_name="dash+strip_False",
        input_text="   hello   world   ",
        expected_text="-hello-world-",
        replacement="-",
        strip=False,
    ),
]


@dataclass(frozen=True, slots=True)
class NormalizeStringTestCase:
    test_name: str
    input_text: str
    expected_text: str
    replacement: str
    strip: bool


TEST_NORMALIZE_STRING: list[NormalizeStringTestCase] = [
    NormalizeStringTestCase(
        test_name="space+strip_True",
        input_text="hello   world",
        expected_text="hello world",
        replacement=" ",
        strip=True,
    ),
    NormalizeStringTestCase(
        test_name="space+strip_False",
        input_text="  hello   world  ",
        expected_text=" hello world ",
        replacement=" ",
        strip=False,
    ),
    NormalizeStringTestCase(
        test_name="empty_string+strip_True",
        input_text="  hello   world  ",
        expected_text="helloworld",
        replacement="",
        strip=True,
    ),
    NormalizeStringTestCase(
        test_name="empty_string+strip_False",
        input_text="  hello   world  ",
        expected_text="helloworld",
        replacement="",
        strip=False,
    ),
    NormalizeStringTestCase(
        test_name="space+strip_True_newline_tab_1",
        input_text=" hi\\n\\tworld ",
        expected_text="hi world",
        replacement=" ",
        strip=True,
    ),
    NormalizeStringTestCase(
        test_name="space+strip_False_newline_tab_1",
        input_text="   hi\\n\\tworld   ",
        expected_text=" hi world ",
        replacement=" ",
        strip=False,
    ),
    NormalizeStringTestCase(
        test_name="empty_string+strip_True_newline_tab_1",
        input_text=" hi\\n\\tworld ",
        expected_text="hiworld",
        replacement="",
        strip=True,
    ),
    NormalizeStringTestCase(
        test_name="empty_string+strip_False_newline_tab_1",
        input_text="  hi\\n\\tworld ",
        expected_text="hiworld",
        replacement="",
        strip=False,
    ),
    NormalizeStringTestCase(
        test_name="space+strip_True_newline_tab_2",
        input_text=" hi\\n\\tworld ",
        expected_text="hi world",
        replacement=" ",
        strip=True,
    ),
    NormalizeStringTestCase(
        test_name="space+strip_False_newline_tab_2",
        input_text="   hi\\n\\tworld   ",
        expected_text=" hi world ",
        replacement=" ",
        strip=False,
    ),
    NormalizeStringTestCase(
        test_name="empty_string+strip_True_newline_tab_2",
        input_text=" hi\\n\\tworld ",
        expected_text="hiworld",
        replacement="",
        strip=True,
    ),
    NormalizeStringTestCase(
        test_name="empty_string+strip_False_newline_tab_2",
        input_text="  hi\\n\\tworld ",
        expected_text="hiworld",
        replacement="",
        strip=False,
    ),
    NormalizeStringTestCase(
        test_name="space+strip_True_newline_tab_3",
        input_text=" hi\\n\\tworld ",
        expected_text="hi world",
        replacement=" ",
        strip=True,
    ),
    NormalizeStringTestCase(
        test_name="space+strip_False_newline_tab_3",
        input_text="   hi\\n\\tworld   ",
        expected_text=" hi world ",
        replacement=" ",
        strip=False,
    ),
    NormalizeStringTestCase(
        test_name="empty_string+strip_True_newline_tab_3",
        input_text=" hi\\n\\tworld ",
        expected_text="hiworld",
        replacement="",
        strip=True,
    ),
    NormalizeStringTestCase(
        test_name="empty_string+strip_False_newline_tab_3",
        input_text="  hi\\n\\tworld ",
        expected_text="hiworld",
        replacement="",
        strip=False,
    ),
    NormalizeStringTestCase(
        test_name="space+strip_True_newline_tab_4",
        input_text=" hi\\n\\tworld ",
        expected_text="hi world",
        replacement=" ",
        strip=True,
    ),
    NormalizeStringTestCase(
        test_name="space+strip_False_newline_tab_4",
        input_text="   hi\\n\\tworld   ",
        expected_text=" hi world ",
        replacement=" ",
        strip=False,
    ),
    NormalizeStringTestCase(
        test_name="empty_string+strip_True_newline_tab_4",
        input_text=" hi\\n\\tworld ",
        expected_text="hiworld",
        replacement="",
        strip=True,
    ),
]


@dataclass(frozen=True, slots=True)
class TokenizeByCharNgramsTestCase:
    test_name: str
    input_text: str
    n: int
    expected_tokens: list[str]


TEST_TOKENIZE_BY_CHAR_NGRAMS: list[TokenizeByCharNgramsTestCase] = [
    TokenizeByCharNgramsTestCase(
        test_name="1_char_ngrams",
        input_text="hello my name",
        n=1,
        expected_tokens=["h", "e", "l", "l", "o", " ", "m", "y", " ", "n", "a", "m", "e"],
    ),
    TokenizeByCharNgramsTestCase(
        test_name="2_char_ngrams",
        input_text="hello my name",
        n=2,
        expected_tokens=["he", "el", "ll", "lo", "o ", " m", "my", "y ", " n", "na", "am", "me"],
    ),
    TokenizeByCharNgramsTestCase(
        test_name="3_char_ngrams",
        input_text="hello my name",
        n=3,
        expected_tokens=["hel", "ell", "llo", "lo ", "o m", " my", "my ", "y n", " na", "nam", "ame"],
    ),
    TokenizeByCharNgramsTestCase(
        test_name="4_char_ngrams",
        input_text="hello my name",
        n=4,
        expected_tokens=["hell", "ello", "llo ", "lo m", "o my", " my ", "my n", "y na", " nam", "name"],
    ),
    TokenizeByCharNgramsTestCase(
        test_name="5_char_ngrams",
        input_text="hello my name",
        n=5,
        expected_tokens=["hello", "ello ", "llo m", "lo my", "o my ", " my n", "my na", "y nam", " name"],
    ),
    TokenizeByCharNgramsTestCase(test_name="empty_string_1_char_ngrams", input_text="", n=1, expected_tokens=[]),
    TokenizeByCharNgramsTestCase(test_name="empty_string_2_char_ngrams", input_text="", n=2, expected_tokens=[]),
    TokenizeByCharNgramsTestCase(test_name="empty_string_3_char_ngrams", input_text="", n=3, expected_tokens=[]),
    TokenizeByCharNgramsTestCase(test_name="empty_string_4_char_ngrams", input_text="", n=4, expected_tokens=[]),
    TokenizeByCharNgramsTestCase(test_name="empty_string_5_char_ngrams", input_text="", n=5, expected_tokens=[]),
    TokenizeByCharNgramsTestCase(test_name="empty_string_1000_char_ngrams", input_text="", n=1000, expected_tokens=[]),
    TokenizeByCharNgramsTestCase(
        test_name="1_char_ngrams_wine", input_text="wine", n=1, expected_tokens=["w", "i", "n", "e"]
    ),
    TokenizeByCharNgramsTestCase(
        test_name="2_char_ngrams_wine", input_text="wine", n=2, expected_tokens=["wi", "in", "ne"]
    ),
    TokenizeByCharNgramsTestCase(
        test_name="3_char_ngrams_wine", input_text="wine", n=3, expected_tokens=["win", "ine"]
    ),
    TokenizeByCharNgramsTestCase(test_name="4_char_ngrams_wine", input_text="wine", n=4, expected_tokens=["wine"]),
    TokenizeByCharNgramsTestCase(test_name="5_char_ngrams_wine", input_text="wine", n=5, expected_tokens=["wine"]),
    TokenizeByCharNgramsTestCase(
        test_name="1000_char_ngrams_wine", input_text="wine", n=1000, expected_tokens=["wine"]
    ),
    TokenizeByCharNgramsTestCase(
        test_name="1_char_ngrams_space", input_text="     ", n=1, expected_tokens=[" ", " ", " ", " ", " "]
    ),
    TokenizeByCharNgramsTestCase(
        test_name="2_char_ngrams_space", input_text="     ", n=2, expected_tokens=["  ", "  ", "  ", "  "]
    ),
    TokenizeByCharNgramsTestCase(
        test_name="3_char_ngrams_space", input_text="     ", n=3, expected_tokens=["   ", "   ", "   "]
    ),
    TokenizeByCharNgramsTestCase(
        test_name="4_char_ngrams_space", input_text="     ", n=4, expected_tokens=["    ", "    "]
    ),
    TokenizeByCharNgramsTestCase(test_name="5_char_ngrams_space", input_text="     ", n=5, expected_tokens=["     "]),
    TokenizeByCharNgramsTestCase(test_name="6_char_ngrams_space", input_text="     ", n=6, expected_tokens=["     "]),
    TokenizeByCharNgramsTestCase(
        test_name="1000_char_ngrams_space", input_text="     ", n=1000, expected_tokens=["     "]
    ),
]


@dataclass(frozen=True, slots=True)
class TokenizeByNcharsTestCase:
    test_name: str
    input_text: str
    n: int
    expected_tokens: list[str]


TEST_TOKENIZE_BY_NCHARS: list[TokenizeByNcharsTestCase] = [
    TokenizeByNcharsTestCase(
        test_name="1_nchars",
        input_text="hello my name",
        n=1,
        expected_tokens=["h", "e", "l", "l", "o", " ", "m", "y", " ", "n", "a", "m", "e"],
    ),
    TokenizeByNcharsTestCase(
        test_name="2_nchars", input_text="hello my name", n=2, expected_tokens=["he", "ll", "o ", "my", " n", "am", "e"]
    ),
    TokenizeByNcharsTestCase(
        test_name="3_nchars", input_text="hello my name", n=3, expected_tokens=["hel", "lo ", "my ", "nam", "e"]
    ),
    TokenizeByNcharsTestCase(
        test_name="4_nchars", input_text="hello my name", n=4, expected_tokens=["hell", "o my", " nam", "e"]
    ),
    TokenizeByNcharsTestCase(
        test_name="5_nchars", input_text="hello my name", n=5, expected_tokens=["hello", " my n", "ame"]
    ),
    TokenizeByNcharsTestCase(
        test_name="1000_nchars", input_text="hello my name", n=1000, expected_tokens=["hello my name"]
    ),
    TokenizeByNcharsTestCase(test_name="empty_string_1_nchars", input_text="", n=1, expected_tokens=[]),
    TokenizeByNcharsTestCase(test_name="empty_string_2_nchars", input_text="", n=2, expected_tokens=[]),
    TokenizeByNcharsTestCase(test_name="empty_string_3_nchars", input_text="", n=3, expected_tokens=[]),
    TokenizeByNcharsTestCase(test_name="empty_string_4_nchars", input_text="", n=4, expected_tokens=[]),
    TokenizeByNcharsTestCase(test_name="empty_string_5_nchars", input_text="", n=5, expected_tokens=[]),
    TokenizeByNcharsTestCase(test_name="empty_string_1000_nchars", input_text="", n=1000, expected_tokens=[]),
    TokenizeByNcharsTestCase(test_name="1_nchars_wine", input_text="wine", n=1, expected_tokens=["w", "i", "n", "e"]),
    TokenizeByNcharsTestCase(test_name="2_nchars_wine", input_text="wine", n=2, expected_tokens=["wi", "ne"]),
    TokenizeByNcharsTestCase(test_name="3_nchars_wine", input_text="wine", n=3, expected_tokens=["win", "e"]),
    TokenizeByNcharsTestCase(test_name="4_nchars_wine", input_text="wine", n=4, expected_tokens=["wine"]),
    TokenizeByNcharsTestCase(test_name="5_nchars_wine", input_text="wine", n=5, expected_tokens=["wine"]),
    TokenizeByNcharsTestCase(test_name="1000_nchars_wine", input_text="wine", n=1000, expected_tokens=["wine"]),
    TokenizeByNcharsTestCase(
        test_name="1_nchars_space", input_text="     ", n=1, expected_tokens=[" ", " ", " ", " ", " "]
    ),
    TokenizeByNcharsTestCase(test_name="2_nchars_space", input_text="     ", n=2, expected_tokens=["  ", "  ", " "]),
    TokenizeByNcharsTestCase(test_name="3_nchars_space", input_text="     ", n=3, expected_tokens=["   ", "  "]),
    TokenizeByNcharsTestCase(test_name="4_nchars_space", input_text="     ", n=4, expected_tokens=["    ", " "]),
    TokenizeByNcharsTestCase(test_name="5_nchars_space", input_text="     ", n=5, expected_tokens=["     "]),
    TokenizeByNcharsTestCase(test_name="6_nchars_space", input_text="     ", n=6, expected_tokens=["     "]),
    TokenizeByNcharsTestCase(test_name="1000_nchars_space", input_text="     ", n=1000, expected_tokens=["     "]),
]


@dataclass(frozen=True, slots=True)
class PreprocessStringTestCase:
    test_name: str
    input_text: str
    expected_text: str
    preprocessing_method: ImplementedStringPreprocessing


TEST_PREPROCESS_STRING: list[PreprocessStringTestCase] = [
    PreprocessStringTestCase(
        test_name="normalize",
        input_text="nice   week",
        expected_text="nice week",
        preprocessing_method=ImplementedStringPreprocessing.NORMALIZE_STRING,
    ),
    PreprocessStringTestCase(
        test_name="normalize_with_padding",
        input_text="   nice   week   ",
        expected_text="nice week",
        preprocessing_method=ImplementedStringPreprocessing.NORMALIZE_STRING,
    ),
    PreprocessStringTestCase(
        test_name="whitespace_removed",
        input_text="123 lol 456",
        expected_text="123lol456",
        preprocessing_method=ImplementedStringPreprocessing.REMOVE_WHITESPACES,
    ),
    PreprocessStringTestCase(
        test_name="whitespace_removed_with_padding",
        input_text="   123 lol 456   ",
        expected_text="123lol456",
        preprocessing_method=ImplementedStringPreprocessing.REMOVE_WHITESPACES,
    ),
    PreprocessStringTestCase(
        test_name="normalize_hello_world",
        input_text="hello   world",
        expected_text="hello world",
        preprocessing_method=ImplementedStringPreprocessing.NORMALIZE_STRING,
    ),
    PreprocessStringTestCase(
        test_name="unicode_escape_newline",
        input_text="hello\\nworld",
        expected_text="hello\nworld",
        preprocessing_method=ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
    ),
    PreprocessStringTestCase(
        test_name="unicode_escape_tab",
        input_text="hello\\tworld",
        expected_text="hello\tworld",
        preprocessing_method=ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
    ),
    PreprocessStringTestCase(
        test_name="unicode_escape_carriage",
        input_text="hello\\rworld",
        expected_text="hello\rworld",
        preprocessing_method=ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
    ),
    PreprocessStringTestCase(
        test_name="replace_multiple_whitespaces",
        input_text="hello   world",
        expected_text="hello world",
        preprocessing_method=ImplementedStringPreprocessing.NORMALIZE_STRING,
    ),
    PreprocessStringTestCase(
        test_name="replace_multiple_whitespaces_with_padding",
        input_text="  hello   world    ",
        expected_text="hello world",
        preprocessing_method=ImplementedStringPreprocessing.NORMALIZE_STRING,
    ),
    PreprocessStringTestCase(
        test_name="whitespace_removed_hello_world",
        input_text="hello   world",
        expected_text="helloworld",
        preprocessing_method=ImplementedStringPreprocessing.REMOVE_WHITESPACES,
    ),
    PreprocessStringTestCase(
        test_name="whitespace_removed_hello_world_with_padding",
        input_text="   hello   world  ",
        expected_text="helloworld",
        preprocessing_method=ImplementedStringPreprocessing.REMOVE_WHITESPACES,
    ),
    PreprocessStringTestCase(
        test_name="unicode_escape_cyrillic",
        input_text="нет привет\nмир",  # noqa: RUF001
        expected_text="нет привет\nмир",  # noqa: RUF001
        preprocessing_method=ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
    ),
    PreprocessStringTestCase(
        test_name="unicode_escape_newline_final",
        input_text="hello\\nworld",
        expected_text="hello\nworld",
        preprocessing_method=ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
    ),
]


@dataclass(frozen=True, slots=True)
class TokenizeStringTestCase:
    test_name: str
    input_text: str
    tokenization_method: ImplementedTokenization
    n: int
    expected_tokens: list[str]


TEST_TOKENIZE_STRING: list[TokenizeStringTestCase] = [
    TokenizeStringTestCase(
        test_name="char_ngrams_3",
        input_text="hello   world",
        tokenization_method=ImplementedTokenization.CHAR_NGRAMS,
        n=3,
        expected_tokens=["hel", "ell", "llo", "lo ", "o  ", "   ", "  w", " wo", "wor", "orl", "rld"],
    ),
    TokenizeStringTestCase(
        test_name="nchars_3",
        input_text="hello   world",
        tokenization_method=ImplementedTokenization.NCHARS,
        n=3,
        expected_tokens=["hel", "lo ", "  w", "orl", "d"],
    ),
    TokenizeStringTestCase(
        test_name="char_ngrams_2",
        input_text="hello my name",
        tokenization_method=ImplementedTokenization.CHAR_NGRAMS,
        n=2,
        expected_tokens=["he", "el", "ll", "lo", "o ", " m", "my", "y ", " n", "na", "am", "me"],
    ),
    TokenizeStringTestCase(
        test_name="nchars_3_hello_my_name",
        input_text="hello my name",
        tokenization_method=ImplementedTokenization.NCHARS,
        n=3,
        expected_tokens=["hel", "lo ", "my ", "nam", "e"],
    ),
]


@dataclass(frozen=True, slots=True)
class PreprocessTestCase:
    test_name: str
    input_text: str
    string_preprocessing_method: ImplementedStringPreprocessing | None
    tokenization_method: ImplementedTokenization | None
    n: int
    expected_after_string_preprocessing: str
    expected_after_tokenization: list[str] | None
    expected_n_param: int | None
    expected_is_tokenized: bool
    expected_result: str | list[str]
    expected_config_name: str


TEST_PREPROCESS_METHOD: list[PreprocessTestCase] = [
    PreprocessTestCase(
        test_name="both_methods_normalize_and_char_ngrams",
        input_text="Hello  World",
        string_preprocessing_method=ImplementedStringPreprocessing.NORMALIZE_WHITESPACES,
        tokenization_method=ImplementedTokenization.CHAR_NGRAMS,
        n=3,
        expected_after_string_preprocessing="Hello World",
        expected_after_tokenization=["Hel", "ell", "llo", "lo ", "o W", " Wo", "Wor", "orl", "rld"],
        expected_n_param=3,
        expected_is_tokenized=True,
        expected_result=["Hel", "ell", "llo", "lo ", "o W", " Wo", "Wor", "orl", "rld"],
        expected_config_name="input_str__str_normalize_whitespaces__tok_char_ngrams__n_3",
    ),
    PreprocessTestCase(
        test_name="both_methods_normalize_and_nchars",
        input_text="Hello  World",
        string_preprocessing_method=ImplementedStringPreprocessing.NORMALIZE_STRING,
        tokenization_method=ImplementedTokenization.NCHARS,
        n=4,
        expected_after_string_preprocessing="Hello World",
        expected_after_tokenization=["Hell", "o Wo", "rld"],
        expected_n_param=4,
        expected_is_tokenized=True,
        expected_result=["Hell", "o Wo", "rld"],
        expected_config_name="input_str__str_normalize_string__tok_nchars__n_4",
    ),
    PreprocessTestCase(
        test_name="string_only_normalize_whitespaces",
        input_text="Hello  World",
        string_preprocessing_method=ImplementedStringPreprocessing.NORMALIZE_WHITESPACES,
        tokenization_method=None,
        n=3,
        expected_after_string_preprocessing="Hello World",
        expected_after_tokenization=None,
        expected_n_param=None,
        expected_is_tokenized=False,
        expected_result="Hello World",
        expected_config_name="input_str__str_normalize_whitespaces__tok_None__n_None",
    ),
    PreprocessTestCase(
        test_name="string_only_remove_whitespaces",
        input_text="Hello  World",
        string_preprocessing_method=ImplementedStringPreprocessing.REMOVE_WHITESPACES,
        tokenization_method=None,
        n=3,
        expected_after_string_preprocessing="HelloWorld",
        expected_after_tokenization=None,
        expected_n_param=None,
        expected_is_tokenized=False,
        expected_result="HelloWorld",
        expected_config_name="input_str__str_remove_whitespaces__tok_None__n_None",
    ),
    PreprocessTestCase(
        test_name="no_string_preprocessing_but_tokenization_char_ngrams",
        input_text=" 1 Hello  World ",
        string_preprocessing_method=None,
        tokenization_method=ImplementedTokenization.NCHARS,
        n=5,
        expected_after_string_preprocessing=" 1 Hello  World ",
        expected_after_tokenization=[" 1 He", "llo  ", "World", " "],
        expected_n_param=5,
        expected_is_tokenized=True,
        expected_result=[" 1 He", "llo  ", "World", " "],
        expected_config_name="input_str__str_None__tok_nchars__n_5",
    ),
    PreprocessTestCase(
        test_name="tokenization_only_char_ngrams",
        input_text="Hello",
        string_preprocessing_method=None,
        tokenization_method=ImplementedTokenization.CHAR_NGRAMS,
        n=2,
        expected_after_string_preprocessing="Hello",
        expected_after_tokenization=["He", "el", "ll", "lo"],
        expected_n_param=2,
        expected_is_tokenized=True,
        expected_result=["He", "el", "ll", "lo"],
        expected_config_name="input_str__str_None__tok_char_ngrams__n_2",
    ),
    PreprocessTestCase(
        test_name="tokenization_only_nchars",
        input_text="Hello",
        string_preprocessing_method=None,
        tokenization_method=ImplementedTokenization.NCHARS,
        n=2,
        expected_after_string_preprocessing="Hello",
        expected_after_tokenization=["He", "ll", "o"],
        expected_n_param=2,
        expected_is_tokenized=True,
        expected_result=["He", "ll", "o"],
        expected_config_name="input_str__str_None__tok_nchars__n_2",
    ),
    PreprocessTestCase(
        test_name="default_n_value",
        input_text="Hello",
        string_preprocessing_method=None,
        tokenization_method=ImplementedTokenization.CHAR_NGRAMS,
        n=3,
        expected_after_string_preprocessing="Hello",
        expected_after_tokenization=["Hel", "ell", "llo"],
        expected_n_param=3,
        expected_is_tokenized=True,
        expected_result=["Hel", "ell", "llo"],
        expected_config_name="input_str__str_None__tok_char_ngrams__n_3",
    ),
    PreprocessTestCase(
        test_name="empty_string_with_string_preprocessing",
        input_text="",
        string_preprocessing_method=ImplementedStringPreprocessing.NORMALIZE_WHITESPACES,
        tokenization_method=None,
        n=3,
        expected_after_string_preprocessing="",
        expected_after_tokenization=None,
        expected_n_param=None,
        expected_is_tokenized=False,
        expected_result="",
        expected_config_name="input_str__str_normalize_whitespaces__tok_None__n_None",
    ),
    PreprocessTestCase(
        test_name="empty_string_with_tokenization",
        input_text="",
        string_preprocessing_method=None,
        tokenization_method=ImplementedTokenization.CHAR_NGRAMS,
        n=3,
        expected_after_string_preprocessing="",
        expected_after_tokenization=[],
        expected_n_param=3,
        expected_is_tokenized=True,
        expected_result=[],
        expected_config_name="input_str__str_None__tok_char_ngrams__n_3",
    ),
    PreprocessTestCase(
        test_name="unicode_escapes_with_tokenization",
        input_text="hello\\nworld",
        string_preprocessing_method=ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES,
        tokenization_method=ImplementedTokenization.CHAR_NGRAMS,
        n=3,
        expected_after_string_preprocessing="hello\nworld",
        expected_after_tokenization=["hel", "ell", "llo", "lo\n", "o\nw", "\nwo", "wor", "orl", "rld"],
        expected_n_param=3,
        expected_is_tokenized=True,
        expected_result=["hel", "ell", "llo", "lo\n", "o\nw", "\nwo", "wor", "orl", "rld"],
        expected_config_name="input_str__str_decode_unicode_escapes__tok_char_ngrams__n_3",
    ),
    PreprocessTestCase(
        test_name="both_methods_none",
        input_text="Hello World",
        string_preprocessing_method=None,
        tokenization_method=None,
        n=3,
        expected_after_string_preprocessing="Hello World",
        expected_after_tokenization=None,
        expected_n_param=None,
        expected_is_tokenized=False,
        expected_result="Hello World",
        expected_config_name="input_str__str_None__tok_None__n_None",
    ),
]
