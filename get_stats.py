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
from functools import partial
from tqdm import tqdm
from collections import Counter
import time


def read_jsonl(fn: str) -> Iterable[Dict[Any, Any]]:
    with open(fn) as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def multi_pool(my_function: Callable, data: List[Dict[Any, Any]],
               n_processes: int, chunk_size: Optional[int],
               functions: List[Callable]) -> List[Any]:
    """
    multi_pool is used to apply the normalizers and filters on one file with
    multiple processes.
    """
    if chunk_size is None:
        chunk_size = min((len(data) // n_processes, 1))

    my_f = partial(my_function, sub_functions=functions)
    with mp.Pool(processes=n_processes) as pool:
        return_list = pool.map(my_f, tqdm(data, total=len(data)), chunksize=chunk_size)
        # return_dict = list(pool.starmap(my_function, tqdm(itertools.product(data, [functions]), total=len(data)), chunksize=chunk_size))
    return return_list


def count(jobj, cnt):
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
    #print(cnt)
    return cnt


def count_updater(cnt, jobj_cnt):
    for key in jobj_cnt:
        for year in jobj_cnt[key]:
            if year not in cnt[key]:
                # print(key, year, jobj_cnt[key][year])
                cnt[key][year] = jobj_cnt[key][year]
                print(key, year, cnt[key][year])
            else:
                # either int then add or list and append
                cnt[key][year] += jobj_cnt[key][year]
                # if type(cnt[key][year]) == int:
                #     cnt[key][year] += jobj_cnt[key][year]
                # elif type(cnt[key][year]) == list:
                #     cnt[key][year] += jobj_cnt[key][year]
    return cnt


def count_wrapper(fn, cnts):
    cnt = {}
    for jobj in read_jsonl(fn):
        jobj_cnt = {
                    "#docs": {},
                    "#words": {},
                    "doc lengths": {},
                    "titles": {},
                    }
        # cnt.update(count(jobj, jobj_cnt))
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
        # cnt = count_updater(cnt, jobj_cnt)
        # print(jobj_cnt["#docs"])
    cnts.append(cnt)
    return cnts


def get_files_from_folders(root, suffix):
    return filter(lambda x: x.endswith(suffix),
                  [os.path.join(f[0], e) for f in os.walk(root) for e in f[2]]
                  )


def main(folder_name, chunksize, n_processes):
    if chunksize is None:
        chunksize = min((len(get_files_from_folders(folder_name, ".jsonl")) // n_processes, 1))
    manager = mp.Manager()
    cnts = manager.list()
    # cnts["#docs"] = {}
    # cnts["#words"] = {}
    # cnts["doc lengths"] = {}
    # cnts["titles"] = {}
    cnt = {
            "#docs": {},
            "#words": {},
            "doc lengths": {},
            "titles": {},
            }
    # my_f = partial(count_wrapper, cnt=cnt)
    # return_dict = pool.map(my_f, get_files_from_folders(folder_name, ".jsonl"), chunksize=chunksize)
    # return_dict = map(my_f, get_files_from_folders(folder_name, ".jsonl"))
    jobs = []
    start = time.time()
    for fn in get_files_from_folders(folder_name, ".jsonl"):
        p = mp.Process(target=count_wrapper, args=(fn, cnts))
        jobs.append(p)
        p.start()
        if len(jobs) > n_processes:
            for proc in jobs:
                proc.join()
            jobs = []
    for proc in jobs:
        proc.join()
    print(time.time() - start)

    print(len(cnts))
    start = time.time()
    for x in cnts:
        for k in x:
            for y in x[k]:
                if y not in cnt[k]:
                    cnt[k][y] = x[k][y]
                else:
                    cnt[k][y] += x[k][y]
    print(time.time() - start)
    print(cnt.keys())
    print(cnt["#docs"])
    print(cnt["#words"])
    print(sum(cnt["doc lengths"]) / len(cnt["doc lengths"]))
    # print(Counter(cnt["titles"]))
    print("done")
    return cnts


def generate_fake_data():
    import lorem
    import random
    os.mkdir("./fake-data")
    os.mkdir("./fake-data/a")
    os.mkdir("./fake-data/b")
    os.mkdir("./fake-data/c")

    template = lambda year, title, content: {
                "meta":
                        {
                            "package_id": "fake123",
                            "title": title,
                            "created": year,
                            "year": year,
                            "edition": 123,
                            "issue": "adsf",
                        },
                "content":
                            content
                }
    for x in "abc":
        for i in tqdm(range(1988, 2022)):
            with open(f"fake-data/{x}/{i}.jsonl", "w") as fout:
                for _ in range(random.randint(1_000, 2_000)):
                    print(json.dumps(template(i, lorem.sentence()[0], [lorem.text() for _ in range(10, 100)])), file=fout)
    return


def simple_main(folder_name):
    cnt = {
            "#docs": {},
            "#words": {},
            "doc lengths": {},
            "titles": {},
            }
    cnts = []
    start = time.time()
    for fn in get_files_from_folders(folder_name, ".jsonl"):
        count_wrapper(fn, cnts)
    print(time.time() - start)

    start = time.time()
    for x in cnts:
        for k in x:
            for y in x[k]:
                if y not in cnt[k]:
                    cnt[k][y] = x[k][y]
                else:
                    cnt[k][y] += x[k][y]
    print(time.time() - start)
    print(cnt.keys())
    print(cnt["#docs"])
    print(cnt["#words"])
    print(sum(cnt["doc lengths"]) / len(cnt["doc lengths"]))
    # print(Counter(cnt["titles"]))
    print("done")
    return cnts


if __name__ == "__main__":
    # generate_fake_data()
    print("multi")
    start = time.time()
    cnt = main("fake-data", 1, 8)
    print(time.time() - start)

    print("single")
    start = time.time()
    cnt = simple_main("fake-data")
    print(time.time() - start)
