import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

@functools.cache
def data_dir() -> Path:
    root_dir = Path(__file__).resolve().parent
    return root_dir / "data"

def load_cpi():
    data_file = data_dir() / "cpi_u_rs.csv"
    values = np.loadtxt(data_file, delimiter=",")[:,1:-1]
    years = np.loadtxt(data_file, dtype=int, delimiter=",", usecols=0)
    out = dict()
    for i, year in enumerate(years):
        for j in range(12):
            out[(year,j+1)] = values[i,j]
    return out

def load_earnings():
    data_file = data_dir() / "earning_percentiles.csv"
    values = np.loadtxt(data_file, delimiter=",")[:,2:]
    times = np.loadtxt(data_file, dtype=int, delimiter=",", usecols=(0,1))
    times = [(row[0], row[1]) for row in times]
    return times, values

def main() -> None:

    # Load earnings data
    times, earnings = load_earnings()

    # Adjust for inflation
    cpi = load_cpi()
    base_cpi = 449.3  # 2023 average
    for i, (year, month) in enumerate(times):
        earnings[i,:] *= base_cpi / cpi[(year, month)]

    # Plot
    x = np.array([year + (month-1) / 12 for year, month in times])
    plt.plot(x, earnings[:, 3])
    plt.show()

if __name__ == "__main__":
    main()
