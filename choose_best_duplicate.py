import json
import re
from typing import Dict, Iterable, Any
from tqdm import tqdm
import difflib
import multiprocessing
import functools
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


# def read_dup_candidates(fn: str) -> Dict[int, Dict[int, float]]:
def read_dup_candidates(fn: str) -> Dict[str, Dict[str, float]]:
    dc = {}
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


# def mirror_dup_candidates(dc: Dict[int, Dict[int, float]]) -> Dict[int, Dict[int, float]]:
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


def get_dduped_text_json(fn: str, fn_out: str, dc: Dict[str, Dict[str, float]], scores: Dict[str, Dict[str, float]]) -> None:
    seen = set()
    keep = {}

    def len_fun(x):
        return 10**math.log(len(x.split()))

    def split_key(key):
        k, i = key.split("_")
        i = int(i)
        return k, i

    key2doc = {k: d for k, d in get_keys_and_docs(fn)}
    total = len(key2doc)

    for doc_key in tqdm(key2doc, total=total):
        if doc_key not in seen:
            doc = key2doc[doc_key]
            k, i = split_key(doc_key)
            # init candidate list
            candidates = [(doc_key, doc, scores[k][i] / len_fun(doc))]
            seen.add(doc_key)
            # doc has duplicates
            if doc_key in dc:
                for j in dc[doc_key].keys():
                    if j not in seen:
                        k, i = split_key(j)
                        candidates.append((j, key2doc[j], scores[k][i] / len_fun(doc)))
                        seen.add(j)
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
                new_docs = list(zip(*sorted(keep[key].items(), key=lambda x: x[0])))[1]
            except KeyError:
                # the content in this element has already been completely filtered out
                new_docs = []
            new_element["content"] = new_docs
            print(json.dumps(new_element), file=fout)


# def get_duplicate_texts(fn: str, fn_out: str, dc: Dict[int, Dict[int, float]]) -> None:
#     start = time.time()
#     with open(fn) as fh:
#         fh.seek(0)
#         content = fh.readlines()
#
#     print("creating managed objects")
#     manager = multiprocessing.Manager()
#     content = manager.list(content)
#     dc = manager.dict(dc)
#     print("starting multiprocess")
#     fout = open(fn_out, "w")
#     with multiprocessing.get_context("spawn").Pool(20) as pool:
#         gd = functools.partial(get_dupes, content=content, dc=dc)
#         results = pool.imap_unordered(gd, tqdm(enumerate(content), total=len(content)), chunksize=len(content) // 20)
#         for r in results:
#             if r:
#                 print(" ".join(r), file=fout)
#     fout.close()
#     print(f"Done! Took {time.time() - start} seconds.")


def get_duplicate_texts(fn: str, fn_out: str, dc: Dict[int, Dict[int, float]], scores: Dict[str, Any]) -> None:
    start = time.time()
    key2doc = {k: d for k, d in get_keys_and_docs(fn)}

    print("creating managed objects")
    manager = multiprocessing.Manager()
    key2doc = manager.dict(key2doc)
    dc = manager.dict(dc)
    print("starting multiprocess")
    fout = open(fn_out, "w")
    with multiprocessing.get_context("spawn").Pool(20) as pool:
        gd = functools.partial(get_dupes, key2doc=key2doc, dc=dc, scores=scores)
        results = pool.imap_unordered(gd, tqdm(key2doc.items(), total=len(key2doc)), chunksize=len(key2doc) // 20 // 20)
        for r in results:
            if r:
                print("\n".join(r), file=fout)
    fout.close()
    print(f"Done! Took {time.time() - start} seconds.")


def get_dupes(key_doc, key2doc, dc, scores):
    result = []
    key, doc = key_doc

    def len_fun(x):
        return 10**math.log(len(x.split()))

    if key in dc:
        kk, ik = key.split("_")
        ik = int(ik)
        # 10.0**(-1 / length)
        len_factor = len_fun(doc)
        result.append(f"KenLM: {scores[kk][ik] / len_factor}\n{doc}\n")
        for j in dc[key].keys():
            kj, ij = j.split("_")
            ij = int(ij)
            len_factor = len_fun(key2doc[j])
            result.append(f"{key}->{j}: {dc[key][j]} KenLM: {scores[kj][ij] / len_factor}\n{diff_strings(doc, key2doc[j])}\n")
            # print(f"{i}->{j}: {dc[i][j]}\n{content[j]}", file=fout)
        result.append("#"*30)
    return result


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--duplicate-candidates", type=str)
    parser.add_argument("--scores", type=str)
    parser.add_argument("--num-workers", type=int, default=20)

    return parser.parse_args()


def main():
    start = time.time()

    args = get_args()

    dc = read_dup_candidates(args.duplicate_candidates)
    edc = mirror_dup_candidates(dc)

    with open(args.scores) as fin:
        score_dict = json.load(fin)

    get_dduped_text_json(args.input, args.output, edc, score_dict)

    print(f"Done :)\nThis took {time.time() - start:,.2f} seconds")


if __name__ == "__main__":
    main()
    # import sys
    # dc = read_dup_candidates(sys.argv[1])
    # print(len(dc))
    # edc = mirror_dup_candidates(dc)
    # with open("tmp/2019.scores.jsonl", "r") as fin:
    #     score_dict = json.load(fin)
    # print(len(dc), len(edc))
    # # get_dduped_text(sys.argv[2], sys.argv[2] + ".dduped", edc)
    # # get_dduped_text_json(sys.argv[2], sys.argv[2] + ".dduped", edc, score_dict)
    # get_duplicate_texts(sys.argv[2], sys.argv[2] + ".duplicates", edc, score_dict)
