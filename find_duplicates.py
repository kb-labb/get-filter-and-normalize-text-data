# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from functools import partial
import json
from lsh import cache, minhash
import multiprocessing
import numpy as np
import time
import pickle
from tqdm import tqdm
import random
from typing import Iterable, Dict, Any, List, Optional, Tuple, Set, BinaryIO


def read_jsonl(fn: str) -> Iterable[Dict[Any, Any]]:
    with open(fn) as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def get_keys_and_docs(fn: str) -> Iterable[Tuple[str, str]]:
    """
    The keys generated here combine the package-id and the position in the list
    of its content.
    This key will in the following be used as a url.
    """
    for jobj in read_jsonl(fn):
        key = jobj["meta"]["package_id"]
        for i, doc in enumerate(jobj["content"]):
            yield f"{key}_{i}", doc


# This function is adapted from:
#   https://github.com/mattilyra/LSH/blob/master/examples/Introduction.ipynb
def shingles(text: str, char_ngram: int = 5) -> Set[str]:
    return set(text[head:head + char_ngram]
               for head in range(0, len(text) - char_ngram))


# This function is adapted from:
#  https://github.com/mattilyra/LSH/blob/master/examples/Introduction.ipynb
def jaccard(set_a: Set[str], set_b: Set[str], args: argparse.Namespace) -> float:
    if len(set_a) < 1 or len(set_b) < 1:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    if args.jaccard == 'min':
        return len(intersection) / min(len(set_a), len(set_b))
    elif args.jaccard == 'max':
        return len(intersection) / max(len(set_a), len(set_b))
    else:
        return len(intersection) / len(union)


def compute_fingerprint(key_text: Tuple[str, str]
                        ) -> Tuple[Optional[str], Optional[str], Any, bool]:
    try:
        key, text = key_text
        fingerprint = hasher.fingerprint(text)
    except Exception as e:
        print('Error:', e)
        return None, None, None, False

    return key, text, fingerprint, True


def url_pairs_to_remove(bucket_urls: List[str], args: argparse.Namespace,
                        url_doc: Dict[str, str]
                        ) -> Tuple[Dict[str, Dict[str, float]], int, int]:
    """
    The main function filtering duplicate documents
    """
    remove_urls_dict = {}
    deduped_local, counter_local = 0, 0
    iteration = 0
    # if len(bucket_urls) < 2:
    #     return remove_urls_dict, deduped_local, counter_local
    while len(bucket_urls) > 1:
        if args.heuristic_iter != -1 and \
                iteration == args.heuristic_iter:
            break

        remove_urls = {}
        main_url = random.choice(tuple(bucket_urls))
        main_shingles = shingles(url_doc[main_url])
        bucket_urls.remove(main_url)

        to_remove = set()
        for other_url in bucket_urls:
            counter_local += 1
            other_shingles = shingles(url_doc[other_url])
            try:
                jaccard_sim = jaccard(main_shingles, other_shingles, args)
            except Exception as e:
                print('Error:', e)
                jaccard_sim = 0.0
            if jaccard_sim > 0.5:
                remove_urls[other_url] = jaccard_sim
                deduped_local += 1
                to_remove.add(other_url)
        for x in to_remove:
            bucket_urls.remove(x)

        if len(remove_urls) > 0:
            remove_urls_dict[main_url] = remove_urls
        iteration += 1
    return remove_urls_dict, deduped_local, counter_local


def write_remove_urls_dict(remove_urls: List[Dict[str, Dict[str, float]]],
                           f_out: BinaryIO) -> None:
    if len(remove_urls) > 0:
        # f_out.write(json.dumps(remove_urls, ensure_ascii=False).encode("utf-8"))
        # f_out.write('\n'.encode('utf-8'))
        for each_url_remove in remove_urls:
            myjson = json.dumps(each_url_remove, ensure_ascii=False)
            f_out.write(myjson.encode('utf-8'))
            f_out.write('\n'.encode('utf-8'))


def url_pairs_to_remove_bin(bin: Dict[str, str], url_doc: Dict[str, str],
                            args: argparse.Namespace
                            ) -> Tuple[List[Dict[str, Dict[str, float]]], int, int]:
    # i = multiprocessing.current_process()._identity[0]
    remove_urls_list = []
    deduped_local, counter_local = 0, 0
    url_ptr = partial(url_pairs_to_remove, args=args, url_doc=url_doc)
    # url_remove_iter = map(url_ptr, tqdm(filter(lambda x: len(x) > 1, bin.values()), position=i))
    url_remove_iter = map(url_ptr, tqdm(filter(lambda x: len(x) > 1, bin.values())))
    # url_remove_iter = map(url_ptr, tqdm(bin.values(), total=len(bin.values()), position=i))
    for rud, dedup, count in url_remove_iter:
        deduped_local += dedup
        counter_local += count
        remove_urls_list.append(rud)

    return remove_urls_list, deduped_local, counter_local


def find_pair_urls_parallel(args: argparse.Namespace, lshcache: cache.Cache,
                            url_doc: Dict[str, str]) -> None:
    start_time = time.time()
    with open(args.output, 'wb') as f_out:
        num_bins = len(lshcache.bins)
        print(f"num_bins: {num_bins}")
        with multiprocessing.get_context("spawn").Pool(num_bins) as pool:
            cj = partial(url_pairs_to_remove_bin, url_doc=url_doc, args=args)
            cji = pool.imap_unordered(cj, lshcache.bins)
            for remove_urls, deduped_local, counter_local in cji:
                write_remove_urls_dict(remove_urls, f_out)
            pool.terminate()

    print(' Taken time for jaccard similarities {:.2f} seconds'.format(
        time.time() - start_time), flush=True)


def find_pair_urls_sequential(args: argparse.Namespace, lshcache: cache.Cache,
                              url_doc: Dict[str, str]) -> None:
    start_time = time.time()
    with open(args.output, 'wb') as f_out:

        num_bins = len(lshcache.bins)
        print(f"num_bins: {num_bins}")
        cj = partial(url_pairs_to_remove_bin, url_doc=url_doc, args=args)
        cji = map(cj, lshcache.bins)
        for remove_urls, deduped_local, counter_local in cji:
            write_remove_urls_dict(remove_urls, f_out)

    print(' Taken time for jaccard similarities {:.2f} seconds'.format(
        time.time() - start_time), flush=True)


def load_fingerprints(args: argparse.Namespace
                      ) -> Tuple[cache.Cache, Dict[str, str]]:
    start = time.time()
    print("Loading...")
    for count_fp, fp_file_name in enumerate(args.load_fingerprints):
        print("Loading fingerprints from pickle file {}".format(
            fp_file_name), flush=True)
        with open(fp_file_name, "rb") as fp:
            if count_fp == 0:
                # assign directory for the first pkl
                lshcache = pickle.load(fp)
                url_doc = pickle.load(fp)
            else:
                # append these to lshcache and url_doc
                local_lshcache = pickle.load(fp)
                local_url_doc = pickle.load(fp)
                for url in local_lshcache.fingerprints.keys():
                    url_doc[url] = local_url_doc[url]
                    lshcache.add_fingerprint(local_lshcache.fingerprints[url], url)
    print(f"Loading took {time.time() - start:.2f} seconds")
    return lshcache, url_doc


def compute_fingerprints(args: argparse.Namespace
                         ) -> Tuple[cache.Cache, Dict[str, str]]:
    print("Computing fingerprints", flush=True)
    for i, input_file in enumerate(args.inputs):
        print(f'document processing {input_file}', flush=True)

        # total = len(list(get_keys_and_docs(input_file)))
        total = sum(1 for _ in get_keys_and_docs(input_file))
        print(f"File {input_file} has {total:,} documents")

        # compute fingerprints in parallel
        with multiprocessing.get_context("fork").Pool(args.num_workers) as pool:
            url_doc = {}
            chunksize = max((1, total // args.num_workers))
            compute_fingerprint_iter = pool.imap_unordered(compute_fingerprint, get_keys_and_docs(input_file), chunksize=chunksize)
            start = time.time()
            for url, text, fingerprint, flag in tqdm(compute_fingerprint_iter, total=total):
                if flag:
                    url_doc[url] = text
                    lshcache.add_fingerprint(fingerprint, url)

            print(f"fingerprinting {input_file} took {time.time() - start:.2f} seconds")
    return lshcache, url_doc


def save_fingerprints(args: argparse.Namespace, lshcache: cache.Cache,
                      url_doc: Dict[str, str]) -> None:
    start = time.time()
    print("Saving fingerprints to pickle file {}".format(
        args.save_fingerprints), flush=True)
    with open(args.save_fingerprints, 'wb') as f_save:
        pickle.dump(lshcache, f_save)
        pickle.dump(url_doc, f_save)
    print(f"Saving took {time.time() - start:.2f} seconds")


def get_args() -> argparse.Namespace:
    print('parsing the arguments ...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed used for python, numpy')
    parser.add_argument('--inputs', nargs='*', default=None,
                        help='List of the input files, ')
    parser.add_argument('--load-fingerprints', nargs='*', default=None,
                        help='Load fingerprints from a list of pickle files,'
                        ' e.g. cc.pkl news.pkl')
    parser.add_argument('--save-fingerprints', type=str, default=None,
                        help='Save the fingerprints of the inputs.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file name that consists of all ids'
                        ' with matching similarities')
    parser.add_argument('--jaccard', type=str, default='union',
                        choices=['union', 'min', 'max'],
                        help='Jaccard similarity computation')
    parser.add_argument('--heuristic-iter', type=int, default=1,
                        help='Number of iterations to run the heuristics: use -1 for exact')
    parser.add_argument('--num-bands', type=int, default=10,
                        help='Number of bands to use in cache')
    parser.add_argument('--num-seeds', type=int, default=100,
                        help='Number of seeds to use for minhash. Note that'
                        ' this value should be divisible by num-bands')
    parser.add_argument('--num-workers', type=int, default=100,
                        help="Number of workers")
    parser.add_argument('--jaccard-parallel', action='store_true',
                        help='Use this to process large number of documents.')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    print('finding possible duplicate content ...')

    # set seed and get an array of seeds of 100 integers
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 1e6, size=args.num_seeds)

    # initialize minhash and lsh cache
    hasher = minhash.MinHasher(seeds=seeds, char_ngram=5, hashbytes=4)
    lshcache = cache.Cache(num_bands=args.num_bands, hasher=hasher)

    # load fingerprints from pickle file if needed
    if args.load_fingerprints is not None:
        lshcache, url_doc = load_fingerprints(args)

    # compute finger prints of the inputs if any
    # input file and the key to use as id
    if args.inputs is not None:
        lshcache, url_doc = compute_fingerprints(args)

    # Save the fingerprints if needed
    if args.save_fingerprints is not None:
        save_fingerprints(args, lshcache, url_doc)

    # compute jaccard index of the input texts and write to file if needed
    if args.output is not None:
        print("Compute jaccard similarity", flush=True)
        if args.jaccard_parallel:
            find_pair_urls_parallel(args, lshcache, url_doc)
        else:
            find_pair_urls_sequential(args, lshcache, url_doc)

    print('done :-)')
