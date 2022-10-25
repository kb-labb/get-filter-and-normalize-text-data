import json
import re
from typing import Dict, Iterable, Any, Optional
from tqdm import tqdm
import difflib
import time
import math
import argparse

from find_duplicates import get_keys_and_docs, read_jsonl


def diff_strings(a: str, b: str, *, use_loguru_colors: bool = False) -> str:
    output = []
    matcher = difflib.SequenceMatcher(None, a, b)
    if use_loguru_colors:
        green = '<GREEN><black>'
        red = '<RED><black>'
        endgreen = '</black></GREEN>'
        endred = '</black></RED>'
    else:
        green = '\x1b[38;5;16;48;5;2m'
        red = '\x1b[38;5;16;48;5;1m'
        endgreen = '\x1b[0m'
        endred = '\x1b[0m'

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            output.append(a[a0:a1])
        elif opcode == 'insert':
            output.append(f'{green}{b[b0:b1]}{endgreen}')
        elif opcode == 'delete':
            output.append(f'{red}{a[a0:a1]}{endred}')
        elif opcode == 'replace':
            output.append(f'{green}{b[b0:b1]}{endgreen}')
            output.append(f'{red}{a[a0:a1]}{endred}')
    return ''.join(output)


def read_jsonl_with_doubles(fn: str) -> Iterable[Any]:
    two_dicts = re.compile(r"\}\{")
    with open(fn) as fh:
        fh_total = sum(1 for _ in fh)
        fh.seek(0)
        for line in tqdm(fh, total=fh_total):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                line = re.sub(two_dicts, "}\n{", line)
                l1, l2 = line.split("\n")
                yield json.loads(l1)
                yield json.loads(l2)


def read_dup_candidates(fn: str) -> Dict[str, Dict[str, float]]:
    dc: Dict[str, Dict[str, float]] = {}
    for jobj in read_jsonl_with_doubles(fn):
        if not jobj:
            continue
        k_id = list(jobj.keys())[0]
        # k_int = int(k_id.split("_")[-1])
        k_int = k_id
        if k_int not in dc:
            dc[k_int] = {}
        for vk, vv in jobj[k_id].items():
            # v_int = int(vk.split("_")[-1])
            v_int = vk
            if v_int not in dc[k_int]:
                dc[k_int][v_int] = vv
            else:
                dc[k_int][v_int] = max(dc[k_int][v_int], vv)  # or min or ...
    return dc


def mirror_dup_candidates(dc: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    edc = {k: {x: y for x, y in v.items()} for k, v in dc.items()}
    for main_key in tqdm(dc):
        for other_key in dc[main_key]:
            if other_key not in edc:
                edc[other_key] = {}
            edc[other_key][main_key] = min(edc[other_key].get(main_key, 0), dc[main_key][other_key])
    return edc


def get_dduped_text(fn: str, fn_out: str, dc: Dict[int, Dict[int, float]]) -> None:
    seen = set()
    with open(fn) as fh, open(fn_out, "w") as fout, open("trash", "w") as trash:
        fh_total = sum(1 for _ in fh)
        fh.seek(0)
        content = fh.readlines()
        fh.seek(0)
        for i, line in tqdm(enumerate(fh), total=fh_total):
            if i not in seen:
                print(line.strip(), file=fout)
                seen.add(i)
                if i in dc:
                    print(line, file=trash)
                    seen.update(dc[i].keys())
                    for j in dc[i].keys():
                        print(content[j], file=trash)


def get_dduped_text_json(fn: str,
                         fn_out: str,
                         dc: Dict[str, Dict[str, float]],
                         scores: Optional[Dict[str, Dict[str, float]]]
                         ) -> None:
    seen = set()
    keep: Dict[str, Dict[int, str]] = {}

    def len_fun(x):
        return 10**math.log(len(x.split()))

    def split_key(key):
        # k, i = key.split("_")
        k_i = key.split("_")
        i = k_i[-1]
        k = "_".join(k_i[:-1])
        i = int(i)
        return k, i

    key2doc: Dict[str, str] = {k: d for (k, d) in get_keys_and_docs(fn)}
    # key2doc: Dict[str, str] = {x[0]: x[1] for x in get_keys_and_docs(fn)}
    total = len(key2doc)

    for doc_key in tqdm(key2doc, total=total):
        if doc_key not in seen:
            doc = key2doc[doc_key]
            k, i = split_key(doc_key)
            # init candidate list
            if scores:
                candidates = [(doc_key, doc, scores[k][i] / len_fun(doc))]
            else:
                candidates = [(doc_key, doc, -len(doc))]
            seen.add(doc_key)
            # doc has duplicates
            if doc_key in dc:
                for j in dc[doc_key].keys():
                    if j not in seen:
                        k, i = split_key(j)
                        if scores:
                            candidates.append((j, key2doc[j], scores[k][i] / len_fun(doc)))
                        else:
                            candidates.append((j, key2doc[j], -len(doc)))
                        seen.add(j)
            # take the doc with minimal score
            # kenlm scores are negative log-likelihoods -> smaller better
            # dividing by the log-length favours longer documents with worse scores
            # as scores for longer texts are smaller/(or bigger in bizarro-log world)
            doc_key, doc, _ = min(candidates, key=lambda x: x[2])
            k, i = split_key(doc_key)
            if k not in keep:
                keep[k] = {}
            keep[k][i] = doc
    with open(fn_out, "w") as fout:
        for element in read_jsonl(fn):
            new_element = element.copy()
            key = new_element["meta"]["package_id"]
            try:
                new_docs = tuple(zip(*sorted(keep[key].items(), key=lambda x: x[0])))[1]
            except KeyError:
                # the content in this element has already been completely filtered out
                new_docs = tuple()
            new_element["content"] = new_docs
            print(json.dumps(new_element), file=fout)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--duplicate-candidates", type=str)
    parser.add_argument("--scores", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    start = time.time()

    args = get_args()

    dc = read_dup_candidates(args.duplicate_candidates)
    edc = mirror_dup_candidates(dc)

    if args.scores:
        with open(args.scores) as fin:
            score_dict = json.load(fin)
    else:
        score_dict = None

    get_dduped_text_json(args.input, args.output, edc, score_dict)

    print(f"Done :)\nThis took {time.time() - start:,.2f} seconds")


if __name__ == "__main__":
    main()
