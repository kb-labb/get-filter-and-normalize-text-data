import os
import argparse
import json
import functools
import data_normalizers as DN
import data_filters as DF
from typing import Callable, List, Iterable, Dict, Any, Set
from tqdm import tqdm
import multiprocessing as mp
import itertools
from functools import partial


def read_jsonl(fn: str) -> Iterable[Dict[Any, Any]]:
    with open(fn) as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def my_normalize(jobj: Dict[Any, Any], sub_functions: List[Callable]) -> Dict[Any, Any]:
    meta = jobj["meta"]
    content = jobj["content"][:]
    for i, c in enumerate(content):
        # content[i] = normalize(content[i])
        for n in sub_functions:
            content[i] = n(content[i])
    return {"meta": meta, "content": content}


def apply_normalizers(fn: str, args: argparse.Namespace) -> None:
    normalizers: List[Callable] = []
    if args.unicode_normalize:
        normalizers.append(DN.unicode_normalize)
    if args.unidecode_normalize:
        normalizers.append(DN.unidecode_normalize)
    if args.moses_normalize:
        normalizers.append(DN.moses_normalize)
    if args.common_errors:
        normalizers.append(DN.common_errors)
    if args.anonymize:
        normalizers.append(DN.anonymize)
    if args.strip_incomplete_string:
        normalizers.append(DN.strip_incomplete_string)
    if args.sentence_split:
        normalizers.append(DN.sentence_split)
    if args.strip_incomplete_sentence:
        if not normalizers[-1] == DN.sentence_split:
            print("--strip_incomplete also require --sentence_split")
            raise Exception("Try again")
        normalizers.append(DN.strip_incomplete_sentence)

    # normalize = compose(*normalizers)

    if normalizers == []:
        return

    data = list(read_jsonl(fn))
    with open(fn + ".normalized", "w") as fout:
        # return_dict = multi_func(my_normalize, data, args.n_processes, 15)
        return_list = multi_pool(my_normalize, data, args.n_processes, 1, normalizers)
        for x in return_list:
            meta = x["meta"]
            content = x["content"]
            print(json.dumps({"meta": meta, "content": content}), file=fout)


def my_filter(jobj: Dict[Any, Any], sub_functions: List[Callable]) -> Dict[Any, Any]:
    meta = jobj["meta"]
    content = jobj["content"][:]
    new_content = []
    for c in content:
        keep = True
        if type(c) == list:  # some docs in SOU are apparently lists
            c = " ".join(c)
        c = c.replace("\n", " ")  # other docs have newlines which some filter does not like
        for f in sub_functions:
            if not f(c):
                keep = False
                break
        if keep:
            new_content.append(c)
    return {"meta": meta, "content": new_content}


def apply_filters(fn: str, args: argparse.Namespace) -> None:
    with mp.Manager() as manager:
        filters: List[Callable] = []
        if args.filter_by_num_tokens:
            filters.append(DF.filter_by_num_tokens)
        if args.filter_by_unicode:
            filters.append(DF.filter_by_unicode)
        if args.filter_tv_tables:
            filters.append(DF.filter_tv_tables)
        # language filter should be run after normalization
        # other filters should be run before to have less data when normalizing
        if args.filter_by_language:
            filters.append(DF.filter_by_language)
        if args.filter_exact_duplicates:
            # hashes: List[int] = [] # manager.list()
            # hashes: Set[int] = set() # manager.list()
            hashes: Dict[int, int] = manager.dict()
            fed = partial(DF.filter_exact_duplicates, hashes=hashes)
            filters.append(fed)

        if filters == []:
            return

        data = list(read_jsonl(fn))

        with open(fn + ".filtered", "w") as fout:
            # return_dict = multi_func(my_filter, data, args.n_processes, None)
            return_list = multi_pool(my_filter, data, args.n_processes, None, filters)
            for x in return_list:
                meta = x["meta"]
                content = x["content"]

                print(json.dumps({"meta": meta, "content": content}), file=fout)


def multi_pool(my_function, data, n_processes, chunk_size, functions):
    if chunk_size is None:
        chunk_size = min((len(data) // n_processes, 1))

    my_f = partial(my_function, sub_functions=functions)
    with mp.Pool(processes=n_processes) as pool:
        return_dict = pool.map(my_f, tqdm(data, total=len(data)), chunksize=chunk_size)
        # return_dict = list(pool.starmap(my_function, tqdm(itertools.product(data, [functions]), total=len(data)), chunksize=chunk_size))
    return return_dict


def json2txt(fn: str) -> None:
    with open(fn.split(".")[0] + ".txt", "w") as fout:
        for jobj in read_jsonl(fn):
            content = jobj["content"]
            for c in content:
                if type(c) == str and c != "":
                    print(c, file=fout)
                elif type(c) == list:
                    # if doc has been split into sentences
                    if c != []:
                        not_empty = False
                        for x in c:
                            if x != "":
                                not_empty = True
                                print(x, file=fout)
                        if not_empty:
                            print("", file=fout)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)

    parser.add_argument("--filter_by_num_tokens", action="store_true")
    parser.add_argument("--filter_by_language", action="store_true")
    parser.add_argument("--filter_by_unicode", action="store_true")
    parser.add_argument("--filter_exact_duplicates", action="store_true")
    parser.add_argument("--filter_tv_tables", action="store_true")

    parser.add_argument("--unicode_normalize", action="store_true")
    parser.add_argument("--unidecode_normalize", action="store_true")
    parser.add_argument("--moses_normalize", action="store_true")
    parser.add_argument("--common_errors", action="store_true")
    parser.add_argument("--anonymize", action="store_true")
    parser.add_argument("--n_processes", type=int, default=1)
    parser.add_argument("--strip_incomplete_string", action="store_true")

    parser.add_argument("--sentence_split", action="store_true")
    parser.add_argument("--strip_incomplete_sentence", action="store_true")

    parser.add_argument("--json2txt", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()
    print(args)
    filter_args = [args.filter_by_unicode, args.filter_exact_duplicates,
                   args.filter_by_language, args.filter_by_num_tokens,
                   args.filter_tv_tables]
    normalize_args = [args.unicode_normalize, args.unidecode_normalize,
                      args.moses_normalize, args.anonymize, args.common_errors,
                      args.strip_incomplete_string,
                      args.strip_incomplete_sentence, args.sentence_split]
    if args.file:
        if any(filter_args):
            apply_filters(args.file, args)
        elif any(normalize_args):
            apply_normalizers(args.file, args)
        elif args.json2txt:
            json2txt(args.file)



    # if args.file:
    #     apply_filters(args.file, args)
    #     apply_normalizers(args.file + ".filtered", args)
    #     if args.json2txt:
    #         json2txt(args.file + ".filtered.normalized")

    # elif args.folder:
    #     files = [os.path.join(args.folder, fn) for fn in os.listdir(args.folder)]
    #     for fn in files:
    #         print(f"Processing {fn}")
    #         try:
    #             apply_filters(fn, args)
    #             apply_normalizers(fn, args)
    #             if args.json2txt:
    #                 json2txt(fn + ".normalized")
    #         except Exception as e:
    #             print(f"Processing {fn} failed with Exception {e}")


if __name__ == "__main__":
    main()
