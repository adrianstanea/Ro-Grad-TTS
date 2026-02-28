from phonemizer.backend import EspeakBackend

from gradtts_ro.symbols import symbols, _symbol_to_id
from gradtts_ro.cleaners import to_lowercase, collapse_whitespace


def create_phonemizer():
    """Create an Espeak backend for Romanian phonemization."""
    return EspeakBackend(
        language='ro',
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
    )


def text_to_phoneme(text, phonemizer):
    """Convert text to phoneme string using the given phonemizer backend."""
    text = to_lowercase(text)
    phonemes = phonemizer.phonemize([text], strip=True)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of phonemes to a sequence of IDs corresponding to the symbols."""
    return [_symbol_to_id[symbol] for symbol in cleaned_text]


def intersperse(lst, item):
    """Adds blank symbol between each element."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result
