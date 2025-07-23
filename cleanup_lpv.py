#lpv data has columns with empty rows in mag_err. this script is meant to touch up any of these .dat files so microlia can run. this can be altered for any type of star.

import os
import pandas as pd

base_dir = "/data01/aschweitzer/software/microlia_output"

#list of filters to loop thru
filters = ["i", "g", "r"]

total_removed = {}

for filt in filters:
    datadir = os.path.join(base_dir, f"training_data_{filt}", "lpv")
    if not os.path.isdir(datadir):
        print(f"Directory not found: {datadir}")
        continue

    files = [f for f in os.listdir(datadir) if f.endswith(".dat")]
    removed = []

    for fname in files:
        fpath = os.path.join(datadir, fname)
        try:
            #read file as space-separated .dat
            df = pd.read_csv(fpath, delim_whitespace=True, header=None)

            #must have exactly 3 columns (time, mag, mag_err in that order)
            if df.shape[1] != 3:
                removed.append(fname)
                os.remove(fpath)
                continue

            #all entries must be numeric and not NAN
            df = df.apply(pd.to_numeric, errors="coerce")
            if df.isnull().any().any():
                removed.append(fname)
                os.remove(fpath)

        except Exception:
            removed.append(fname)
            os.remove(fpath)

    total_removed[filt] = removed
    print(f"[{filt}] Removed {len(removed)} corrupted .dat files.")

#how many files removed
print("\nSummary of removed files:")
for filt, files in total_removed.items():
    for f in files:
        print(f"{filt}: {f}")
