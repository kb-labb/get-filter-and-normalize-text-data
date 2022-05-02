import fasttext
import re
from filter_by_unicode import NICE_CHARS
from typing import Pattern, List, Set, Dict


MODEL = fasttext.load_model("lid.176.bin")


def filter_by_num_tokens(doc: str, min_size: int = 30) -> bool:
    if len(doc.split()) < min_size:
        return False
    else:
        return True


def filter_by_language(doc: str,
                       model: fasttext.FastText._FastText = MODEL,
                       min_prob: float = 0.5) -> bool:

    doc = doc.replace("\n", " ")
    prediction = model.predict(doc, k=2)
    if not (prediction[0][0] != "__label__sv" or
            prediction[1][0] - prediction[1][1] < min_prob):
        return True
    else:
        return False


def filter_by_unicode(doc: str,
                      disallowed_chars: Pattern[str] = NICE_CHARS) -> bool:
    matches = tuple(re.finditer(disallowed_chars, doc))
    if matches:
        return False
    else:
        return True


def filter_tableau(doc: str) -> bool:
    return True


# def filter_exact_duplicates(doc: str, hashes: Set[int]) -> bool:
def filter_exact_duplicates(doc: str, hashes: Dict[int, int]) -> bool:
    # does probably not work properly due to multiprocessing
    h = hash(doc)
    if h not in hashes:
        hashes[h] = 1
        # hashes.append(h)
        # hashes.add(h)
        return True
    return False


def filter_tv_tables(doc: str, n_times: int = 3) -> bool:
    regex = re.compile(r"(\d)?\d(\.|:)\d\d")
    matches = regex.findall(doc)
    if len(matches) > n_times:
        return False
    else:
        return True
