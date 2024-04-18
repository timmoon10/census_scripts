from collections.abc import Iterable

import numpy as np
import pandas
import tqdm

from utils import data_dir

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
    year_data["hourslw"].fillna(0, inplace=True)
    year_data["earnwt"].fillna(0, inplace=True)

    # Split into monthly data
    return [
        year_data[year_data["intmonth"] == month]
        for month in range(1, 13)
    ]

def work_hour_fractions(
    data: pandas.DataFrame,
    work_hour_ranges: Iterable[tuple[float,float]],
) -> list[float]:
    """Fraction of people working a given number of hours"""

    # Sort data by hours worked
    data = data[["hourslw", "earnwt"]].to_numpy(dtype=np.double)
    data = data[data[:,0].argsort()]

    # Cumulative sum of population weights
    weights_cumsum = np.cumsum(data[:,1])

    # Find earning percentiles
    out = []
    for hour_low, hour_high in work_hour_ranges:
        idx_low = np.searchsorted(data[:,0], hour_low)
        idx_low = min(idx_low, data.shape[0]-1)
        idx_high = np.searchsorted(data[:,0], hour_high)
        idx_high = min(idx_high, data.shape[0]-1)
        population = weights_cumsum[idx_high] - weights_cumsum[idx_low]
        out.append(population / weights_cumsum[-1])
    return out

def main() -> None:

    # Options
    ranges = [(1, 200), (20, 200), (40, 200)]  # Work hours
    result_file = data_dir() / "work_hours.csv"

    # Compute earning percentiles
    data = []
    for year in tqdm.tqdm(range(1979, 2024)):
        year_data = load_data(year)
        for month, month_data in enumerate(year_data):
            month += 1
            fractions = work_hour_fractions(month_data, ranges)
            data.append([year, month] + list(fractions))

    # Save results to file
    with open(result_file, "w") as f:
        f.write("# Fraction of population working a given number of hours, from NBER MORG\n")
        f.write("# Year,Month,")
        f.write(",".join(f"{low}-{high}" for low, high in ranges))
        f.write("\n")
        for month_data in data:
            f.write(",".join(str(val) for val in month_data))
            f.write("\n"),

if __name__ == "__main__":
    main()
