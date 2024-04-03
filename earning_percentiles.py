import collections
from collections.abc import Iterable
import functools
from pathlib import Path

import numpy as np
import pandas
import tqdm

@functools.cache
def data_dir() -> Path:
    """Path to project data directory"""
    root_dir = Path(__file__).resolve().parent
    return root_dir / "data"

def load_data(year: int) -> list[pandas.DataFrame]:
    """Load MORG data for a year and split into monthly data"""

    # Load MORG data file
    data_file = data_dir() / "morg" / f"morg{year % 100:0>2}.dta"
    year_data = pandas.read_stata(
        data_file,
        convert_categoricals=False,
    )

    # Filter out non-working age ranges
    year_data = year_data[year_data["age"] >= 25]
    year_data = year_data[year_data["age"] <= 60]

    # Clean data
    year_data = year_data[year_data["year"] == year]
    year_data["earnwke"].fillna(0, inplace=True)
    year_data["earnwt"].fillna(0, inplace=True)

    # Split into monthly data
    return [
        year_data[year_data["intmonth"] == month]
        for month in range(1, 13)
    ]

def distribute_household_earnings(data: pandas.DataFrame) -> pandas.DataFrame:
    """Distribute weekly earnings within households

    Adjust each individual's weekly earnings to the average earnings
    in their household.

    """

    # Compute total household incomes
    households = data["hhid"].to_numpy(dtype=str)
    personal_earnings = data["earnwke"].to_numpy(dtype=np.double)
    household_earnings = collections.defaultdict(lambda: (0.0, 0))
    for i in range(households.size):
        earnings, count = household_earnings[households[i]]
        household_earnings[households[i]] = (
            earnings + personal_earnings[i],
            count + 1,
        )

    # Compute average income within each household
    household_earnings = {
        household: earnings / count
        for household, (earnings, count) in household_earnings.items()
    }

    # Set personal weekly earnings to household average
    personal_earnings = [
        household_earnings[households[i]]
        for i in range(households.size)
    ]
    return data.assign(earnwke=personal_earnings)

def mean_earnings(data: pandas.DataFrame) -> float:
    """Mean weekly earnings"""
    data = data[["earnwke", "earnwt"]].to_numpy(dtype=np.double)
    return data.prod(axis=1).sum() / data[:,1].sum()

def percentile_earnings(
    data: pandas.DataFrame,
    fractions: Iterable[float],
) -> list[float]:
    """Weekly earning percentiles"""

    # Sort data by earnings
    data = data[["earnwke", "earnwt"]].to_numpy(dtype=np.double)
    data = data[data[:,0].argsort()]

    # Cumulative sum of population weights
    weights_cumsum = np.cumsum(data[:,1])

    # Find earning percentiles
    out = []
    for fraction in fractions:
        idx = np.searchsorted(
            weights_cumsum,
            fraction * weights_cumsum[-1],
        )
        out.append(data[idx, 0])
    return out

def main() -> None:

    # Options
    fractions = (0.1, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75, 0.9)  # Earning percentiles
    result_file = data_dir() / "earning_percentiles.csv"

    # Compute earning percentiles
    data = []
    for year in tqdm.tqdm(range(1979, 2024)):
        year_data = load_data(year)
        for month, month_data in enumerate(year_data):
            month += 1
            mean = mean_earnings(month_data)
            percentiles = percentile_earnings(month_data, fractions)
            data.append([year, month, mean] + list(percentiles))

    # Save results to file
    with open(result_file, "w") as f:
        f.write("# Weekly earning percentiles from NBER MORG\n")
        f.write("# Year,Month,Mean,")
        f.write(",".join(str(f) for f in fractions))
        f.write("\n")
        for month_data in data:
            f.write(",".join(str(val) for val in month_data))
            f.write("\n")

if __name__ == "__main__":
    main()
