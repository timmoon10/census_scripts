import argparse
import collections
import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

@functools.cache
def data_dir() -> Path:
    """Path to project data directory"""
    root_dir = Path(__file__).resolve().parent
    return root_dir / "data"

def load_cpi() -> dict[tuple[int, int], float]:
    """Get monthly CPI

    Returns a dictionary where the keys are tuples of the year and
    month.

    """
    data_file = data_dir() / "cpi_u_rs.csv"
    values = np.loadtxt(data_file, delimiter=",")[:,1:-1]
    years = np.loadtxt(data_file, dtype=int, delimiter=",", usecols=0)
    out = dict()
    for i, year in enumerate(years):
        for j in range(12):
            out[(year,j+1)] = values[i,j]
    return out

def load_earnings(
    data_file: Path,
    *,
    col: int,
) -> dict[tuple[int, int], float]:
    """Get monthly nominal earnings

    Column is one-indexed. Returns a dictionary where the keys are
    tuples of the year and month.

    """
    data_file = data_dir() / data_file
    values = np.loadtxt(
        data_file,
        dtype=np.double,
        delimiter=",",
        usecols=col-1,
    )
    times = np.loadtxt(
        data_file,
        dtype=np.int64,
        delimiter=",",
        usecols=(0,1),
    )
    return { (time[0], time[1]): val for time, val in zip(times, values) }

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--title", type=str, default="Weekly earnings", help="Plot title",
    )
    parser.add_argument(
        "--col", type=int, default=6, help="Column to plot (one-indexed)",
    )
    parser.add_argument(
        "--file", type=str, default="earning_percentiles.csv",
        help=f"Earnings data file in {data_dir()}",
    )
    parser.add_argument(
        "--monthly", action="store_true",
        help="Plot monthly data instead of yearly",
    )
    args = parser.parse_args()
    print('\n'.join(f'{key}: {val}' for key, val in vars(args).items()))
    return args

def main() -> None:

    # Command-line arguments
    args = parse_args()

    # Load earnings data
    month_earnings = load_earnings(data_dir() / args.file, col=args.col)

    # Adjust for inflation
    cpi = load_cpi()
    base_cpi = 449.3  # 2023 average
    for i, key in enumerate(month_earnings.keys()):
        month_earnings[key] *= base_cpi / cpi[key]

    # Construct plot data
    if args.monthly:
        pairs = sorted(list(month_earnings.items()))
        x = np.array([year + (month-1) / 12 for (year, month), _ in pairs])
        y = np.array([val for _, val in pairs])
    else:
        year_earnings = collections.defaultdict(lambda: (0.0, 0))
        for (year, month), val in month_earnings.items():
            acc, count = year_earnings[year]
            year_earnings[year] = (acc + val, count + 1)
        year_earnings = {
            year: acc / count
            for year, (acc, count) in year_earnings.items()
        }
        pairs = sorted(list(year_earnings.items()))
        x = np.array([year for year, _ in pairs])
        y = np.array([val for _, val in pairs])

    # Plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(
        xlabel="Year",
        ylabel="Inflation-adjusted 2023 dollars",
        title=args.title,
    )
    ax.set_xlim(round(x[0]), round(x[-1]))
    ax.set_ylim(0, (max(y) // 200) * 300)
    plt.show()

if __name__ == "__main__":
    main()
