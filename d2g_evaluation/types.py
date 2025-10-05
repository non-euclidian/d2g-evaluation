from enum import StrEnum, unique


@unique
class InputFormat(StrEnum):
    STRING = "string"
    TOKEN = "token"
    STRING_AND_TOKEN = "string_and_token"
