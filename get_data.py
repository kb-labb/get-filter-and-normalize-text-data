import os
import multiprocessing as mp
import json
import yaml
import argparse
import itertools as it
import kblab
from typing import TypedDict, List, Dict, Optional, Any, Tuple
import urllib3 as ul3
# from urllib3 import PoolManager, make_headers
# from urllib3.util import Retry

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


def get_file(url: str, package_id: str, fn: str,
             headers) -> Optional[Dict[str, Any]]:

    http = ul3.PoolManager()
    try:
        file = http.request(
            "GET",
            f"{url}{package_id}/{fn}",
            headers=headers,
            retries=ul3.Retry(connect=5,
                              read=4,
                              redirect=5,
                              backoff_factor=0.02),
        )

        return json.loads(file.data.decode("utf-8"))
    except Exception as e:
        print(
            f"Collecting the file {fn} from package {package_id} for url {url} failed with Exception {e}"
        )
        return None


def get_metadata(package_id: str, headers: Dict[str, str], url: str) -> Meta:
    """
    gets metadata
    """
    file = get_file(url, package_id, "meta.json", headers)
    if file is not None:
        try:
            year = int(file.get("year", -1))
        except ValueError:
            year = -1
        return {
            "package_id": package_id,
            "title": file.get("title", "no title"),
            "created": file.get("created", "no created"),
            "year": year,
            "edition": file.get("edition", "no edition"),
            "issue": file.get("issue", "no issue"),
        }
    else:
        print("Failed with package-id {}".format(package_id))
        return {
            "package_id": package_id,
            "title": "failed",
            "created": "failed",
            "year": -1,
            "edition": "failed",
            "issue": "failed",
        }


def get_contentdata(package_id: str, headers: Dict[str, str],
                    url: str) -> List[str]:
    """
    gets content as list of list of string
    """
    file = get_file(url, package_id, "content.json", headers)
    if file:
        content = [x["content"] for x in file]
        return content
    else:
        print("Failed with package-id {}".format(package_id))
        return []


def get_data(package_id: str, url: str, headers: Dict[str, str],
             fpath: str) -> None:
    """
    gets metadata and contentdata and writes them to jsonl-file
    """

    meta = get_metadata(package_id, headers, url)
    content = get_contentdata(package_id, headers, url)
    if meta["title"] == "failed":
        return None
    # title = " ".join(meta["title"].split()[:-1])
    title = "_".join(meta["title"].split()[:-1][:5])
    year = str(meta["year"])
    # fn = os.path.join(fpath, title, f"{year}.jsonl")
    fn = os.path.join(fpath, f"{year}.jsonl")
    # os.makedirs(os.path.join(fpath, title), exist_ok=True)
    os.makedirs(os.path.join(fpath), exist_ok=True)

    with open(fn, "a") as fh:
        print(json.dumps({"meta": meta, "content": content}), file=fh)
    return None


def load_archive(config: Dict[str, str]) -> kblab.Archive:
    return kblab.Archive(config["url"],
                         auth=(config["user"], config["password"]))


def load_config(login_file: str) -> Dict[str, str]:
    with open(login_file) as fh:
        return yaml.load(fh, Loader=yaml.SafeLoader)


def archive_search(archive: kblab.Archive, search_dict: Dict[str, str],
                   start_end: Tuple[int, int]) -> kblab.Result:
    start, end = start_end
    return archive.search(search_dict, start=start, max=end)


def get_ids(archive: kblab.Archive, search_dict: Dict[str, str]) -> List[str]:
    m = archive.search(search_dict).m
    b = 10_000
    # with mp.Pool(processes=40) as pool:
    #     package_ids = pool.starmap(archive_search,
    #                                it.product([archive], [search_dict],
    #                                           zip(range(0, m - b, b),
    #                                               range(b, m, b))),
    #                                chunksize=5000)
    # does not work because mp wants to pickle Archive
    package_ids = []
    for start, end in zip(range(0, m - b, b), range(b, m, b)):
        print(start, end, search_dict)
        package_ids.extend(archive.search(search_dict, start=start, max=min(end, m)))
    return package_ids


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--login_file")
    parser.add_argument("--tag")
    parser.add_argument("--location", default="data")

    return parser.parse_args()


def main() -> None:
    args = get_args()

    config = load_config(args.login_file)
    archive = load_archive(config)

    fpath = os.path.join(args.location, args.tag)
    os.makedirs(fpath, exist_ok=True)

    # package_ids = [x for x in archive.search({"tags": args.tag})]
    package_ids = []
    counter = 0
    my_search = archive.search({"tags": args.tag})
    total = my_search.n
    while True:
        try:
            pid = next(my_search.keys)
            if pid:
                counter += 1
                package_ids.append(pid)
            else:
                break

            if counter % 10_000:
                print("{:.2%}".format(counter / total), end="\r")

        except ul3.exceptions.ProtocolError as e:
            print("\noopsie Protocol")
            print(e)
            continue
        except StopIteration:
            print("\noopsie StopIteration")
            print(len(package_ids), total)
            break

    # package_ids = get_ids(archive, {"tags": args.tag})
    print(f"found {len(package_ids)} packages")

    pw = config["password"]
    headers = ul3.make_headers(basic_auth=f"demo:{pw}")
    print(type(headers))
    print(headers)

    print(archive)
    with mp.Pool(processes=40) as pool:
        c = 0
        for _ in pool.starmap(get_data,
                              it.product(package_ids, [config["url"]],
                                         [headers], [fpath]),
                              chunksize=5000):
            c += 1
            print(c, end="\r")


if __name__ == "__main__":

    main()
