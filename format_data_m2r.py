import json
from typing import Iterable, List, Tuple, TypedDict, Dict, Union, Any, TextIO
import argparse


Meta = TypedDict(
    "Meta",
    {
        "package_id": str,
        "title": str,
        "created": str,  # date
        "year": int,
        "edition": str,  # int
        "issue": str  # int
    })

"""
## START ## /data/ess/package_instance/c6/69/69/29/31a87a5fb2ff696c5d6feffd3e634367
"""


def read_file(fn: str) -> Iterable[Tuple[str, str, List[str]]]:
    with open(fn) as fh:
        text: List[str] = []
        uid: str = ""
        url: str = ""
        for line in fh:
            if line.startswith("## START ##"):
                if text:
                    yield (uid, url, text)
                    text = []
                _line = line.strip().split()
                uid = _line[3]
                if len(_line) > 4:
                    url = _line[4]
                else:
                    url = "failed"
            elif line.strip():
                text.append(line.strip())
        yield (uid, url, text)


{
    "package_id": "failed",
    "title": "failed",
    "created": "failed",
    "year": -1,
    "edition": "failed",
    "issue": "failed",
}


def tuple_to_dict(mytuple: Tuple[str, str, List[str]]
                  ) -> Dict[str, Union[Meta, List[str]]]:
    uid, url, content = mytuple
    meta: Meta = {"package_id": uid,
                  "title": url,
                  "created": "failed",
                  "year": -1,
                  "edition": "failed",
                  "issue": "failed",
                  }
    return {"meta": meta, "content": content}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)

    return parser.parse_args()


def main() -> None:
    args = get_args()

    with open(args.output, "w") as fout:
        for element in read_file(args.input):
            print(json.dumps(tuple_to_dict(element)), file=fout)


if __name__ == "__main__":
    main()
