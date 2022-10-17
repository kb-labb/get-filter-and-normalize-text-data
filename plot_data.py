import json
import matplotlib.pyplot as plt
import argparse


def bar_plot(fn: str, key: str) -> None:
    with open(fn) as fh:
        content = json.load(fh)

    to_plot = {}
    for year in content[key]:
        to_plot.update(content[key][year])
    
    my_bins = [2**i for i in range(1,13)]
    binned_data = {b: 0 for b in my_bins}
    curr_bin = 0
    for x in sorted(to_plot.keys(), key=lambda x: int(x)):
        if int(x) >= my_bins[curr_bin] and curr_bin < len(my_bins) - 1:
            curr_bin += 1
        binned_data[my_bins[curr_bin]] += to_plot[x]

    # plt.bar(to_plot.keys(), to_plot.values())
    plt.bar([str(x) for x in binned_data.keys()], binned_data.values())
    plt.show()
    return


def bar_plot_weighted(fn: str, key: str) -> None:
    with open(fn) as fh:
        content = json.load(fh)

    to_plot = {}
    for year in content[key]:
        to_plot.update(content[key][year])
    
    my_bins = [2**i for i in range(1,13)]
    binned_data = {b: 0 for b in my_bins}
    curr_bin = 0
    for x in sorted(to_plot.keys(), key=lambda x: int(x)):
        if int(x) >= my_bins[curr_bin] and curr_bin < len(my_bins) - 1:
            curr_bin += 1
        binned_data[my_bins[curr_bin]] += to_plot[x] * int(x)

    # plt.bar(to_plot.keys(), to_plot.values())
    plt.bar([str(x) for x in binned_data.keys()], binned_data.values())
    plt.show()
    return


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    parser.add_argument("--key", type=str, default="doc lengths")

    return parser.parse_args()


def main() -> None:
    args = get_args()
    bar_plot(args.filename, args.key)
    bar_plot_weighted(args.filename, args.key)

    return


if __name__ == "__main__":
    main()
