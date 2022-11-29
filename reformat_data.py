import json
from typing import Iterable, List, Tuple, TypedDict, Dict, Union
import argparse
from tqdm import tqdm
from clean_data import read_jsonl


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


def read_file_martin(fn: str) -> Iterable[Tuple[str, str, str, List[str]]]:
    with open(fn) as fh:
        text: List[str] = []
        uid: str = ""
        url: str = ""
        time: str = "failed"
        for line in fh:
            if line.startswith("## START ##"):
                if text:
                    yield (uid, url, time, text)
                    text = []
                _line = line.strip().split()
                uid = "KB_" + _line[3]
                if len(_line) > 4:
                    url = _line[4]
                else:
                    url = "failed"
            elif line.strip():
                text.append(line.strip())
        yield (uid, url, time, text)


def read_file_mc4(fn: str) -> Iterable[Tuple[str, str, str, List[str]]]:
    with open(fn) as fh:
        for line in fh:
            jline = json.loads(line)
            text = jline["text"].split("\n")
            time = jline["timestamp"]
            url = jline["url"]
            yield (f"mc4_{url}-{time}", url, time, text)


def read_file_oscar(fn: str) -> Iterable[Tuple[str, str, str, List[str]]]:
    with open(fn) as fh:
        for line in fh:
            jline = json.loads(line)
            text = jline["content"].split("\n")
            time = jline["warc_headers"]["warc-date"]
            url = jline["warc_headers"]["warc-target-uri"]
            yield (f"oscar_{url}-{time}", url, time, text)



{
    "package_id": "failed",
    "title": "failed",
    "created": "failed",
    "year": -1,
    "edition": "failed",
    "issue": "failed",
}


def tuple_to_dict(mytuple: Tuple[str, str, str, List[str]]
                  ) -> Dict[str, Union[Meta, List[str]]]:
    year = -1
    uid, url, time, content = mytuple
    if url != "failed" and time == "failed":
        uid = f"{uid}_{url}"
    if time != "failed":
        year = int(time.split("-")[0])
    meta: Meta = {"package_id": uid,
                  "title": url,
                  "created": time,
                  "year": year,
                  "edition": "failed",
                  "issue": "failed",
                  }
    return {"meta": meta, "content": content}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--style", default=None, type=str, choices=["martin", "mc4", "oscar"])

    return parser.parse_args()


def main() -> None:
    args = get_args()

    if args.style is None:
        raise Exception("No style given --style")

    if args.style == "martin":
        read_file = read_file_martin
    elif args.style == "mc4":
        read_file = read_file_mc4
    elif args.style == "oscar":
        read_file = read_file_oscar

    with open(args.output, "w") as fout:
        for element in tqdm(read_file(args.input)):
            print(json.dumps(tuple_to_dict(element)), file=fout)


if __name__ == "__main__":
    main()
