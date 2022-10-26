from tqdm import tqdm
import json


def removed(fn_in):
    with open(fn_in) as fin:
        data = json.load(fin)
    for key, value in tqdm(data.items()):
        with open(f"{fn_in}.{key}", "w") as fout:
            for x in value:
                print(x, file=fout)


if __name__ == "__main__":
    import sys
    removed(sys.argv[1])