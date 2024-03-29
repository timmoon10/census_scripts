import collections
from collections.abc import Iterable
import functools
from pathlib import Path

import numpy as np
import pandas

@functools.cache
def data_dir() -> Path:
    root_dir = Path(__file__).resolve().parent
    return root_dir / "data"

def load_data(year: int) -> dict[int, pandas.DataFrame]:

    # Load MORG data file
    data_file = data_dir() / "morg" / f"morg{year % 100:0>2}.dta"
    year_data = pandas.read_stata(
        data_file,
        convert_categoricals=False,
    )

    # Filter out non-working age ranges
    year_data = year_data[year_data["age"] >= 16]
    year_data = year_data[year_data["age"] <= 65]

    # Clean data
    year_data = year_data[year_data["year"] == year]
    year_data["earnwke"].fillna(0, inplace=True)
    year_data["earnwt"].fillna(0, inplace=True)


    # Extract per-month data
    out = dict()
    col_labels = list(year_data.columns)
    household_col = col_labels.index("hhid")
    earnings_col = col_labels.index("earnwke")
    for month in range(1, 13):
        month_data = year_data[year_data["intmonth"] == month]

        # Distribute income within households
        households = month_data["hhid"]
        household_earnings = collections.defaultdict(lambda: (0.0, 0))
        for i in range(month_data.shape[0]):
            household = households.iat[i]
            earnings, count = household_earnings[household]
            earnings += month_data.iat[i, earnings_col]
            count += 1
            household_earnings[household] = (earnings, count)
        household_earnings = {
            household: earnings / count
            for household, (earnings, count) in household_earnings.items()
        }
        for i in range(month_data.shape[0]):
            household = households.iat[i]
            month_data.iat[i, earnings_col] = household_earnings[household]

        out[month] = month_data

    return out

def mean_earnings(data: pandas.DataFrame) -> float:
    data = data[["earnwke", "earnwt"]].to_numpy(dtype=np.double)
    return data.prod(axis=1).sum() / data[:,1].sum()

def percentile_earnings(
    data: pandas.DataFrame,
    fractions: Iterable[float],
) -> list[tuple[float, float]]:

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
        out.append((fraction, data[idx, 0]))
    return out

def main() -> None:

    # Earning percentiles
    fractions = (0.1, 0.25, 0.5, 0.75, 0.9)

    # Print data for all months
    print("# Weekly earning percentiles from NBER MORG")
    print("# Year, Month, Mean, " + ", ".join(str(f) for f in fractions))
    for year in range(1979, 2024):
        data = load_data(year)
        for month in range(1, 13):
            earnings = percentile_earnings(data[month], fractions)
            print(
                f"{year}, {month}, {mean_earnings(data[month])}, "
                + ", ".join(f"{earn}" for _, earn in earnings)
            )

if __name__ == "__main__":
    main()
