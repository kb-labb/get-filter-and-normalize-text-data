import json
from clean_data import read_jsonl
from typing import Tuple, Dict, List
from get_data import Meta
from tqdm import tqdm
import argparse


def change_id(jobj: Dict[str, Tuple[Meta, List[str]]],
              id_prefix: str,
              delimiter: str,
              ) -> Dict[str, Tuple[Meta, List[str]]]:
    new_id = id_prefix + delimiter + jobj["meta"]["package_id"]
    jobj["meta"]["package_id"] = new_id
    return jobj


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--infile", type=str, default=None)
    parser.add_argument("--outfile", type=str, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--delimiter", type=str, default="_")

    return parser.parse_args()


def main():
    args = get_args()

    if args.infile is None or args.outfile is None or args.prefix is None:
        raise Exception("Please put some arguments")

    with open(args.outfile, "w") as fout:
        for jobj in tqdm(read_jsonl(args.infile)):
            print(json.dumps(change_id(jobj, args.prefix, args.delimiter)), file=fout)


if __name__ == "__main__":
    main()
