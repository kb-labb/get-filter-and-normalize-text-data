import re
import json
from sentence_splitter import SentenceSplitter
from unidecode import unidecode  # this requires the hacked version
from tokenizers.normalizers import NFKC, Normalizer
from string_replacer import StringReplacer, StringReplacerLite
from sacremoses.normalize import MosesPunctNormalizer
from typing import List


with open("replace.json") as fh:
    ERROR_MAP = json.load(fh)
ERROR_REPLACER = StringReplacer(ERROR_MAP)

ANON_MAP = {
        # ersätt person- och samordningsnummer med XXXXXX - XXXX
        r'(19|20)?[0-9]{2}(01|02|03|04|05|06|07|08|09|10|11|12)[0-9]{2}[0-9]{4}': 'XXXXXXXXXX',
        r'(19|20)?[0-9]{2}(01|02|03|04|05|06|07|08|09|10|11|12)[0-9]{2} - [0-9]{4}': 'XXXXXX - XXXX',
        r'(19|20)?[0-9]{2}(01|02|03|04|05|06|07|08|09|10|11|12)[0-9]{2}-[0-9]{4}': 'XXXXXX-XXXX',
        # ersätt mailadresser
        r'[\w.]+@\w+(\.\w+)+': 'MEJLADRESS',
        r'[\w]+( \. \w+)* @ \w+( \. \w+)+': 'MEJLADRESS',
        # ersätt nicks på formen @nick
        r'@\w+': 'NICK',
        r'@ \w+': 'NICK'
    }
# Telephone numbers?
ANONYMIZER = StringReplacerLite(ANON_MAP)

MOSES_NORMALIZER = MosesPunctNormalizer("sv")

SPLITTER = SentenceSplitter(language="sv")


def unicode_normalize(doc: str,
                      normalizer: Normalizer = NFKC()) -> str:
    """
    This script uses a chosen tokenizers normalizer.
    The default normalizer is a simple NFKC
    """
    return normalizer.normalize_str(doc)


def unidecode_normalize(doc: str) -> str:
    """
    This script uses a modified unidecode to normalize all kinds of characters.
    This is a pretty brutal method getting rid of many nuances.
    ÖÄÅ... are saved from this sledgehammer method.
    """
    return unidecode(doc)


def moses_normalize(doc: str) -> str:
    """
    Compared to unidecode this might be less intrusive as it only deals with
    a select few punctuation marks.
    """
    return MOSES_NORMALIZER.normalize(doc)


def common_errors(doc: str) -> str:
    return ERROR_REPLACER(doc)


def anonymize(doc: str) -> str:
    return ANONYMIZER(doc)


def _anonymize(doc: str) -> str:
    _doc = doc
    for pattern, replacement in ANON_MAP.items():
        _doc = re.sub(pattern, replacement, _doc)
    return _doc


# def strip_incomplete_string(doc: str) -> str:
#     # the first version of the regex checks that docs do not start with a lowercase letter
#     # beginning = re.compile(r"^([A-ZÖÄÅØÆ]|\W|\d)")
#     # this version checks that the doc starts with a capital letter
#     beginning = re.compile(r"^([A-ZÖÄÅØÆ])")
#     end = re.compile(r"([.!?\"\'])$")
#
#     doc = doc.split()
#
#     while len(doc) > 0 and beginning.search(doc[0]) is None:
#         doc = doc[1:]
#     while len(doc) > 0 and end.search(doc[-1]) is None:
#         doc = doc[:-1]
#     return " ".join(doc)

def strip_incomplete_string(doc: str) -> str:
    beginning = re.compile(r"([A-ZÖÄÅØÆ]\w+)")
    end = re.compile(r"([.!?\"\'])\s")
    try:
        a = beginning.search(doc).start()
    except AttributeError:
        a = 0
    try:
        b = list(end.finditer(doc))[-1].start() + 1
    except IndexError:
        b = -1
    return doc[a:b]


def strip_incomplete_sentence(doc: List[str]) -> List[str]:
    # the first version of the regex checks that docs do not start with a lowercase letter
    # beginning = re.compile(r"^([A-ZÖÄÅØÆ]|\W|\d)")
    # this version checks that the doc starts with a capital letter
    beginning = re.compile(r"^([A-ZÖÄÅØÆ])")
    end = re.compile(r"([.!?\"\'])$")

    while len(doc) > 0 and beginning.search(doc[0]) is None:
        doc = doc[1:]
    while len(doc) > 0 and end.search(doc[-1]) is None:
        doc = doc[:-1]
    return doc


def sentence_split(doc: str) -> List[str]:
    return SPLITTER.split(text=doc)


def add_forgotten_spaces(doc: str) -> str:
    # Detär nuett Dåkomprissar arenan.Och vindflöjarna.- plötsligtlåta
    pass


if __name__ == "__main__":
    import time

    n = int(1e6)
    start = time.time()
    (anonymize(n*"19001233-1234 oder 20221212 - 0987 oder first.last@mail.com aber auch @xyz123 und @ 123"))
    print(time.time() - start)
    start = time.time()
    (_anonymize(n*"19001233-1234 oder 20221212 - 0987 oder first.last@mail.com aber auch @xyz123 und @ 123"))
    print(time.time() - start)
