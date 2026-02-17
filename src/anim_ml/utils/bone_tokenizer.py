from __future__ import annotations

TOKEN_LENGTH = 32
PAD_TOKEN = 0
UNK_TOKEN = 1

CHAR_TO_TOKEN: dict[str, int] = {
    "_": 2,
    ".": 3,
    "-": 4,
    ":": 5,
    "/": 6,
    " ": 7,
    "a": 8, "b": 9, "c": 10, "d": 11, "e": 12, "f": 13,
    "g": 14, "h": 15, "i": 16, "j": 17, "k": 18, "l": 19,
    "m": 20, "n": 21, "o": 22, "p": 23, "q": 24, "r": 25,
    "s": 26, "t": 27, "u": 28, "v": 29, "w": 30, "x": 31,
    "y": 32, "z": 33,
    "0": 34, "1": 35, "2": 36, "3": 37, "4": 38,
    "5": 39, "6": 40, "7": 41, "8": 42, "9": 43,
}

VOCAB_SIZE = 64


def tokenize_bone_name(name: str) -> list[int]:
    lower = name.lower()
    tokens = [CHAR_TO_TOKEN.get(ch, UNK_TOKEN) for ch in lower]

    if len(tokens) > TOKEN_LENGTH:
        tokens = tokens[-TOKEN_LENGTH:]

    padding = [PAD_TOKEN] * (TOKEN_LENGTH - len(tokens))
    return padding + tokens
