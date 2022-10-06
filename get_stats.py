# "meta":
#         {
#             "package_id": package_id,
#             "title": file.get("title", "no title"),
#             "created": file.get("created", "no created"),
#             "year": year,
#             "edition": file.get("edition", "no edition"),
#             "issue": file.get("issue", "no issue"),
#         },
# "content":
#             list(str) oder list(list(str))


import os
import json
import multiprocessing as mp
from typing import List, Dict, Any, Callable, Optional, Iterable
from tqdm import tqdm
import time
import argparse


CNT = Dict[str, Dict[int, Any]]
JOBJ = Dict[Any, Any]


def read_jsonl(fn: str) -> Iterable[JOBJ]:
    with open(fn) as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def count(jobj: Dict[str, Any], cnt: CNT) -> CNT:
    meta = jobj["meta"]
    year = meta["year"]
    title = meta["title"]
    content = jobj["content"]

    for c in content:
        if type(c) == list:  # if it is sentence-split just ignore that for now
            c = " ".join(c)
        len_c = len(c.split())
        if len_c > 0:
            if year not in cnt["#docs"]:
                cnt["#docs"][year] = 0
            if year not in cnt["#words"]:
                cnt["#words"][year] = 0
            if year not in cnt["doc lengths"]:
                cnt["doc lengths"][year] = []
            if year not in cnt["titles"]:
                cnt["titles"][year] = []

            cnt["#docs"][year] += 1
            cnt["#words"][year] += len_c
            cnt["doc lengths"][year].append(len_c)
            cnt["titles"][year].append(title)
    return cnt


def count_updater(cnt: CNT, jobj_cnt: CNT) -> CNT:
    for key in jobj_cnt:
        for year in jobj_cnt[key]:
            if year not in cnt[key]:
                cnt[key][year] = jobj_cnt[key][year]
                print(key, year, cnt[key][year])
            else:
                # either int then add or list and append
                cnt[key][year] += jobj_cnt[key][year]
    return cnt


def count_wrapper(fn: str, cnts: List[CNT]) -> List[CNT]:
    cnt = {}
    for jobj in read_jsonl(fn):
        jobj_cnt: CNT = {
                        "#docs": {},
                        "#words": {},
                        "doc lengths": {},
                        "titles": {},
                        }
        jobj_cnt = count(jobj, jobj_cnt)
        for k in jobj_cnt:
            if k not in cnt:
                cnt[k] = jobj_cnt[k]
            else:
                for y in jobj_cnt[k]:
                    if y not in cnt[k]:
                        cnt[k][y] = jobj_cnt[k][y]
                    else:
                        cnt[k][y] += jobj_cnt[k][y]
    cnts.append(cnt)
    return cnts


def get_files_from_folders(root: str, suffix: str) -> Iterable[str]:
    return filter(lambda x: x.endswith(suffix),
                  [os.path.join(f[0], e) for f in os.walk(root) for e in f[2]]
                  )


def print_counts(cnt: CNT) -> None:
    for year in sorted(cnt["#docs"]):
        avg_doc_len = sum(cnt["doc lengths"][year]) / len(cnt["doc lengths"][year])
        nwords = cnt["#words"][year]
        ndocs = cnt["#docs"][year]
        print(f"{year:<10} {nwords:>20,} {ndocs:>20,} {avg_doc_len:>10.2f}")
        # print(Counter(cnt["titles"][year]))
    return


def main(folder_name: str,
         n_processes: int,
         file_suffix: str,
         outfile: str
         ) -> None:
    manager = mp.Manager()
    cnts = manager.list()
    cnt: CNT = {
                "#docs": {},
                "#words": {},
                "doc lengths": {},
                "titles": {},
                }
    jobs = []
    start = time.time()
    n_files = len(list(get_files_from_folders(folder_name, file_suffix)))
    for fn in tqdm(get_files_from_folders(folder_name, file_suffix), total=n_files):
        p = mp.Process(target=count_wrapper, args=(fn, cnts))
        jobs.append(p)
        p.start()
        if len(jobs) > n_processes:
            for proc in jobs:
                proc.join()
            jobs = []
    for proc in jobs:
        proc.join()
    print(f"processing files took {time.time() - start:.2f} seconds")

    start = time.time()
    for x in cnts:
        for k in x:
            for y in x[k]:
                if y not in cnt[k]:
                    cnt[k][y] = x[k][y]
                else:
                    cnt[k][y] += x[k][y]
    print(f"collecting counts per file to one large counter took {time.time() - start:.2f} seconds")

    print_counts(cnt)

    with open(outfile, "w") as fout:
        json.dump(cnt, fout)

    print("done")


def get_args() -> argparse.Namespace:
    print('parsing the arguments ...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-processes', type=int, default=8,
                        help="number of processes used for counting files in parallel")
    parser.add_argument('--folder', type=str, default=None,
                        help="folder that contains the files that need counting")
    parser.add_argument('--suffix', type=str, default=None,
                        help="file-suffix to filter which files need to be counted")
    parser.add_argument('--output', type=str, default=None,
                        help="file name to write the report to")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args.folder, args.n_processes, args.suffix, args.output)
