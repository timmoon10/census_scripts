import argparse
import collections
from collections.abc import Iterable
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from utils import data_dir, results_dir

def load_work_hour_percentage_data(
    data_file: pathlib.Path,
    *,
    cols: Iterable[int],
) -> dict[tuple[int, int], float]:
    """Get monthly employment rates

    Column is one-indexed. Returns a dictionary where the keys are
    tuples of the year and month.

    """
    data = np.loadtxt(
        data_file,
        dtype=np.double,
        delimiter=",",
        usecols=[col-1 for col in cols],
    )
    data *= 100  # Convert to percentages
    times = np.loadtxt(
        data_file,
        dtype=np.int64,
        delimiter=",",
        usecols=(0,1),
    )
    return { (time[0], time[1]): vals for time, vals in zip(times, data) }

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--title", type=str, default="Employment rate", help="Plot title",
    )
    parser.add_argument(
        "--cols", type=int, nargs="+", default=[5],
        help="Columns to plot (one-indexed)",
    )
    parser.add_argument(
        "--data-file", type=str, default="work_hours.csv",
        help=f"Work hour data file in {data_dir()}",
    )
    parser.add_argument(
        "--plot-file", type=str, default=None,
        help=f"Output plot to image file in {results_dir()} instead of showing",
    )
    parser.add_argument(
        "--monthly", action="store_true",
        help="Plot monthly data instead of yearly",
    )
    parser.add_argument(
        "--legend", type=str, nargs="+", default=[],
        help="Legend labels",
    )
    parser.add_argument(
        "--print-years", type=int, nargs="+", default=[],
        help="Print results for specific years",
    )
    args = parser.parse_args()
    print('\n'.join(f'{key}: {val}' for key, val in vars(args).items()))
    return args

def main() -> None:

    # Command-line arguments
    args = parse_args()

    # Load work hour data
    month_percents = load_work_hour_percentage_data(
        data_dir() / args.data_file,
        cols=args.cols,
    )

    # Compute yearly work hour data
    year_percents = collections.defaultdict(
        lambda: (0.0, np.zeros(len(args.cols)))
    )
    for (year, month), vals in month_percents.items():
        acc, count = year_percents[year]
        year_percents[year] = (acc + vals, count + 1)
    year_percents = {
        year: acc / count
        for year, (acc, count) in year_percents.items()
    }

    # Print yearly data
    if args.print_years:
        print("# Column," + ",".join(str(year) for year in args.print_years))
        for i in range(len(args.cols)):
            name = args.legend[i] if args.legend else str(args.cols[i])
            data = [year_percents[year][i] for year in args.print_years]
            print(name + "," + ",".join(f"{val:0.4f}" for val in data))

    # Construct plot data
    if args.monthly:
        pairs = sorted(list(month_percents.items()))
        x = np.array([year + (month-1) / 12 for (year, month), _ in pairs])
        ys = [
            np.array([vals[i] for _, vals in pairs])
            for i in range(len(args.cols))
        ]
    else:
        pairs = sorted(list(year_percents.items()))
        x = np.array([year for year, _ in pairs])
        ys = [
            np.array([vals[i] for _, vals in pairs])
            for i in range(len(args.cols))
        ]

    # Construct plot
    fig, ax = plt.subplots()
    for i, y in enumerate(ys):
        line, = ax.plot(x, y)
        if args.legend:
            line.set_label(args.legend[i])
    ax.set(
        xlabel="Year",
        ylabel="Percent",
        title=args.title,
    )
    ax.set_xlim(math.floor(x[0]), math.ceil(x[-1]))
    ax.set_ylim(0, 100)
    if args.legend:
        ax.legend()

    # Save image file or show plot
    if args.plot_file:
        plt.savefig(results_dir() / args.plot_file)
    else:
        plt.show()

if __name__ == "__main__":
    main()
