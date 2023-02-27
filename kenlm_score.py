from transformers import PreTrainedTokenizerFast
import kenlm
import multiprocessing as mp
import functools
from find_duplicates import get_keys_and_docs, read_jsonl
from tqdm import tqdm
import json
import operator
from typing import Dict, Any, TypeVar, TypedDict, List, Tuple
from sentence_splitter import SentenceSplitter
import argparse
import time

A = TypeVar("A")

Doc = TypedDict("Doc", {"id": str, "doc": List[str], "score": float})


def pp(log_score: float, length: int) -> float:
    return 10.0**(-log_score / length)


# def score_doc(doc: List[str], model: kenlm.Model) -> float:
#     length = sum(len(sen.split()) for sen in doc)
#     if doc:
#         return round(pp(functools.reduce(operator.add, (model.score(s) for s in doc)), length), 1)
#     return -1


def load_tokenizer(path: str, unk: str, mask: str, pad: str, bos: str, eos: str
                   ) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
    tokenizer.bos_token = bos
    tokenizer.eos_token = eos
    tokenizer.cls_token = bos
    tokenizer.sep_token = eos
    tokenizer.mask_token = mask
    tokenizer.unk_token = unk
    tokenizer.pad_token = pad
    return tokenizer


def tok_and_ken(key_text: Tuple[str, str], tokenizer: PreTrainedTokenizerFast,
                lm: kenlm.Model) -> Tuple[str, float]:
    key, text = key_text
    if not text:
        return key, -float("inf")
    tok_list = tokenizer.tokenize(text)
    tok = " ".join(tok_list)
    score = pp(lm.score(tok), len(tok_list))
    return key, score


def tok_split_and_ken(key_text: Tuple[str, str],
                      tokenizer: PreTrainedTokenizerFast, lm: kenlm.Model,
                      splitter: SentenceSplitter) -> Tuple[str, float]:
    key, text = key_text
    sentences = splitter.split(text=text)
    tok_sens = [" ".join(tokenizer.tokenize(sen)) for sen in sentences]
    # tok = " ".join(tok_list)
    score = score_doc(tok_sens, lm)
    return key, score


def score_doc(doc: List[str], model: kenlm.Model) -> float:
    length = sum(len(sen.split()) for sen in doc)
    if doc:
        return round(pp(functools.reduce(operator.add, (model.score(s) for s in doc)), length), 1)
    return -1


def filter_by_score(doc: Dict[str, Any], scores: Dict[str, List[float]]) -> Dict[str, Any]:
    new_content = []
    new_scores = []
    cutoff_head = 430
    # cutoff_tail = 680
    key = doc["meta"]["package_id"]
    for text, score in zip(doc["content"], scores[key]):
        if score < cutoff_head:
            new_content.append(text)
            new_scores.append(score)
    doc["content"] = new_content
    doc["scores"] = new_scores
    return doc


def main1():
    tokenizer = load_tokenizer("tokenizer.json", "[UNK]", "[MASK]", "[PAD]", "[BOS]", "[EOS]")
    lm = kenlm.LanguageModel("tmp/oscar+wiki.ssplit.arpa.bin")

    n_procs = 20
    input_file = "tmp/2019.jsonl.filtered.normalized.filtered.normalized.filtered"

    SPLITTER = SentenceSplitter(language="sv")

    score_dict = {}
    total = 0
    for element in read_jsonl(input_file):
        key = element["meta"]["package_id"]
        docs = element["content"]
        total += len(docs)
        score_dict[key] = [None] * len(docs)
    print(f"File {input_file} has {total:,} documents")

    with mp.get_context("fork").Pool(n_procs) as pool:
        # funky = functools.partial(tok_and_ken, tokenizer=tokenizer, lm=lm)
        funky = functools.partial(tok_split_and_ken, tokenizer=tokenizer, lm=lm, splitter=SPLITTER)
        results = pool.imap_unordered(funky, get_keys_and_docs(input_file), chunksize=total // n_procs)
        for key, score in tqdm(results, total=total):
            k1, k2 = key.split("_")
            k2 = int(k2)
            score_dict[k1][k2] = score
    fails = 0
    for k in score_dict:
        if any(x is None for x in score_dict[k]):
            fails += 1
    print(fails)
    with open("tmp/2019.scores.jsonl", "w") as fout:
        json.dump(score_dict, fout)


def main2():
    input_file = "tmp/2019.jsonl.filtered.normalized.filtered.normalized.filtered"

    total = 0
    for element in read_jsonl(input_file):
        # key = element["meta"]["package_id"]
        docs = element["content"]
        total += len(docs)
    print(f"File {input_file} has {total:,} documents")

    with open("tmp/2019.scores.jsonl", "r") as fin:
        score_dict = json.load(fin)

    with open("tmp/2019.kenlm.jsonl", "w") as fout:
        funky = functools.partial(filter_by_score, scores=score_dict)
        # results = pool.imap_unordered(funky, read_jsonl(input_file), chunksize=total // n_procs)
        results = map(funky, read_jsonl(input_file))
        for element in tqdm(results):
            print(json.dumps(element), file=fout)
            # print(element["meta"]["package_id"])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--kenlm-model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--num-workers", type=int, default=20)

    return parser.parse_args()


def main():
    start = time.time()
    args = get_args()

    tokenizer = load_tokenizer(args.tokenizer, "[UNK]", "[MASK]", "[PAD]", "[BOS]", "[EOS]")
    lm = kenlm.LanguageModel(args.kenlm_model)

    n_procs = args.num_workers
    input_file = args.input
    output_file = args.output

    score_dict = {}
    total = 0
    for element in read_jsonl(input_file):
        key = element["meta"]["package_id"]
        docs = element["content"]
        total += len(docs)
        score_dict[key] = [None] * len(docs)
    print(f"File {input_file} has {total:,} documents")

    with mp.get_context("fork").Pool(n_procs) as pool:
        funky = functools.partial(tok_and_ken, tokenizer=tokenizer, lm=lm)
        chunksize = max((1, total // n_procs))
        results = pool.imap_unordered(funky, get_keys_and_docs(input_file), chunksize=chunksize)
        for key, score in tqdm(results, total=total):
            k1, k2 = key.split("_")
            k2 = int(k2)
            score_dict[k1][k2] = score

    with open(output_file, "w") as fout:
        json.dump(score_dict, fout)

    print(f"Done scoring all documents in {input_file} saving to {output_file}")
    print(f"Tokenizer: {args.tokenizer}, KenLM: {args.kenlm_model}")
    print(f"This took {time.time() - start:,.2f} seconds")


if __name__ == "__main__":
    main()
