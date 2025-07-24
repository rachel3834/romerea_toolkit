import os
import pandas as pd


#filter ALL STARS


base_dir = "/data01/aschweitzer/software/microlia_output"

filters = ["i", "g", "r"]
stars = ["lpv", "ml", "rrlyr", "const"]

total_removed = {}

for filt in filters:
    for star in stars:
        datadir = os.path.join(base_dir, f"training_data_{filt}", star)
        if not os.path.isdir(datadir):
            print(f"Directory not found: {datadir}")
            continue

        files = [f for f in os.listdir(datadir) if f.endswith(".dat")]
        removed = []

        for fname in files:
            fpath = os.path.join(datadir, fname)
            try:
                df = pd.read_csv(fpath, delim_whitespace=True, header=None)


                if df.shape[1] != 3:
                    removed.append(fname)
                    os.remove(fpath)
                    continue


                df = df.apply(pd.to_numeric, errors="coerce")
                if df.isnull().any().any():
                    removed.append(fname)
                    os.remove(fpath)
                    continue


                if (df[2] <= 0.0).any():
                    removed.append(fname)
                    os.remove(fpath)

            except Exception:
                removed.append(fname)
                os.remove(fpath)

        key = f"{filt}_{star}"
        total_removed[key] = removed
        print(f"[{key}] Removed {len(removed)} corrupted .dat files.")


print("\nSummary of removed files:")
for key, files in total_removed.items():
    for f in files:
        print(f"{key}: {f}")
