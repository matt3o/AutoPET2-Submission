# import csv
from __future__ import annotations

# import numpy as np
import sys

# import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.figsize"] = [15, 3]


if __name__ == "__main__":
    assert len(sys.argv) == 4
    folder = sys.argv[1]
    experiment = sys.argv[2]
    # MB or GB as input
    MB_or_GB = sys.argv[3]
    # os.path.basename(os.path.normpath(sys.argv[1]))

    # folder = '../../../../data'
    # experiment = '134'
    folder = f"{folder}/{experiment}/"
    location = f"{folder}/usage.csv"
    df = pd.read_csv(location)

    df["time"] = pd.to_datetime(df["time"])
    start = df["time"].iloc[0]
    end = df["time"].iloc[-1]

    with open(f"{folder}/{experiment}_usage.txt", "w") as usage_file:
        usage_file.write(f"start: {start}, end: {end}\n\n")
        usage_file.write(str(df.describe()))

    try:
        gpu_util = df.loc[:, "gpu util (%)"].rolling(10).sum() / 10
    except KeyError:
        gpu_util = df.loc[:, "utilization"].rolling(10).sum() / 10
        gpu_util.plot(subplots=True, figsize=(15, 2), ylim=(0, 1))

    plot, ax = plt.subplots(figsize=(15, 3))
    ax.set_ylim((0, 1))
    ax.plot(gpu_util, label="gpu util (%)")
    plot.legend()
    plot.savefig(f"{folder}/{experiment}_gpu_util.png")

    try:
        memory_util = df.loc[:, "memory util (%)"].rolling(10).sum() / 10
        plot, ax = plt.subplots(figsize=(15, 3))
        ax.set_ylim((0, 1))
        ax.plot(memory_util, label="memory util (%)")
        plot.legend()
        plot.savefig(f"{folder}/{experiment}_gpu_memory_util.png")
        # memory_util.plot(subplots=True, figsize=(15,2),ylim=(0,1))
    except KeyError:
        pass

    divisor = int(df.shape[0] / 10)
    df["used memory moving average"] = df.loc[:, f"used memory ({MB_or_GB})"].rolling(divisor).sum() / divisor

    plot, ax = plt.subplots()
    from_tick = 0
    to_tick = df.shape[0]
    for i in (f"used memory ({MB_or_GB})", "used memory moving average"):
        ax.plot(df.loc[from_tick:to_tick, i], label=f"df {i}")
        # ax.plot(df2.loc[from_tick:to_tick, 'time'], df2.loc[from_tick:to_tick, i], label=f"df2 {i}")
    # ax.set_xticks(df["time"][:])

    plot.legend()
    plot.savefig(f"{folder}/{experiment}_gpu_usage.png")
