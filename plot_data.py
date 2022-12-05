import json
import matplotlib.pyplot as plt
import argparse


def my_plot(fn: str,
             key: str,
             percentages: bool = False,
             weighted: bool = False,
             pie: bool = False,
             ) -> None:
    with open(fn) as fh:
        content = json.load(fh)

    to_plot = {}
    for year in content[key]:
        to_plot.update(content[key][year])

    my_bins = [2**i for i in range(1, 13)]
    binned_data = {b: 0 for b in my_bins}
    curr_bin = 0
    for x in sorted(to_plot.keys(), key=lambda x: int(x)):
        if int(x) >= my_bins[curr_bin] and curr_bin < len(my_bins) - 1:
            curr_bin += 1
        if weighted:
            binned_data[my_bins[curr_bin]] += to_plot[x] * int(x)
        else:
            binned_data[my_bins[curr_bin]] += to_plot[x]

    # plt.bar(to_plot.keys(), to_plot.values())
    if percentages:
        total = sum(binned_data.values())
        for k, v in binned_data.items():
            binned_data[k] = v / total
    a = "greater than"
    b = "occurences"
    print(f"{a:<15}{b:>20}")
    for k, v in binned_data.items():
        print(f"{k:<15}{v:>20,}")
    if pie:
        # plt.pie(labels=[str(x) for x in binned_data.keys()],
        #         x=binned_data.values(),
        #         autopct="%1.1f%%")
        plt.pie(labels=[(k, f"{v:,}") for k, v in binned_data.items()],
                x=binned_data.values(),
                autopct="%1.1f%%")
    else:
        plt.bar([str(x) for x in binned_data.keys()], binned_data.values())

    pfn = fn[:-5]
    piebar = "pie" if pie else "bar"
    weight = "weighted_docs" if weighted else "raw_docs"
    plt.title(f"{pfn}.{piebar}.{weight}")
    plt.savefig(f"{pfn}.{piebar}.{weight}.svg")
    plt.show()
    return


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    parser.add_argument("--key", type=str, default="doc lengths")

    return parser.parse_args()


def main() -> None:
    args = get_args()
    my_plot(args.filename, args.key)
    my_plot(args.filename, args.key, weighted=True)
    my_plot(args.filename, args.key, pie=True)
    my_plot(args.filename, args.key, pie=True, weighted=True)

    return


if __name__ == "__main__":
    main()
