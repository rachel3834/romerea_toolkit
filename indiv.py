import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import csv
from astropy.io import fits
hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
crossmatch_path = "/data01/aschweitzer/data/ROME/ROME-FIELD-20/ROME-FIELD-20_field_crossmatch.fits"
info_files = {
    "rp": "CV_Lightcurves/Const_fits/variability_std_mean_rp.txt",
    "gp": "CV_Lightcurves/Const_fits/variability_std_mean_gp.txt",
    "ip": "CV_Lightcurves/Const_fits/variability_std_mean_ip.txt"
}
output_dir = "CV_Lightcurves/Const_fits/Indiv_LCs_By_Filter"
os.makedirs(output_dir, exist_ok=True)

Mag_col = 7       # normalized_mag
Err_col = 8       # normalized_mag_err
Time_col = 0      # HJD
QC_col = 16       # QC flag

with h5py.File(hdf5_path, "r") as f:
    raw_data = f["dataset_photometry"][:]
    print("Loaded photometry data:", raw_data.shape)
    num_obs = raw_data.shape[1]

#getting filter to sort
with fits.open(crossmatch_path) as hdul:
    images_table = hdul["IMAGES"].data
    filters = np.char.lower(np.array(images_table["filter"], dtype=str))  # Ensure lowercase for consistency

if len(filters) != num_obs:
    raise ValueError(f"Filter list length {len(filters)} does not match number of observations {num_obs}")

#loading txt summary files
def load_info(path):
    idxs, is_vars = [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t", skipinitialspace=True)
        for row in reader:
            idxs.append(int(row["star_index"]))
            is_vars.append(row["is_variable"].strip().lower() == "true")
    return np.array(idxs), np.array(is_vars)

#now processing summary info files
star_info = {}
for flt, path in info_files.items():
    star_info[flt] = load_info(path)

#picking 1 const and 1 var star that appears in ALL THREE FILTERS
common_consts = set(star_info["rp"][0][~star_info["rp"][1]]) \
    & set(star_info["gp"][0][~star_info["gp"][1]]) \
    & set(star_info["ip"][0][~star_info["ip"][1]])

common_vars = set(star_info["rp"][0][star_info["rp"][1]]) \
    & set(star_info["gp"][0][star_info["gp"][1]]) \
    & set(star_info["ip"][0][star_info["ip"][1]])

if not common_consts or not common_vars:
    raise ValueError("Couldn'ty find stars present as constant/variable in all filters.")

const_idx = random.choice(list(common_consts))
var_idx = random.choice(list(common_vars))

selected_ids = {"constant": const_idx, "variable": var_idx}

for label, sid in selected_ids.items():
    print(f"\nPlotting {label} star: index {sid}")
    star_data = raw_data[sid]
    good = star_data[:, QC_col] == 0

    for flt in ["rp", "gp", "ip"]:
        mask = (filters == flt) & good
        time = star_data[mask, Time_col]
        mag = star_data[mask, Mag_col]
        err = star_data[mask, Err_col]

        if len(time) == 0:
            print(f"No data for star {sid} in {flt} filter.")
            continue

        plt.figure(figsize=(8, 4))
        plt.errorbar(time, mag, yerr=err, fmt="o", ms=4, alpha=0.7, label=f"{label} in {flt}")
        plt.gca().invert_yaxis()
        plt.xlabel("HJD")
        plt.ylabel("Normalized Magnitude")
        plt.title(f"{label.capitalize()} Star (Index {sid}) in {flt} filter")
        plt.tight_layout()

        fname = f"{label}_star_{sid}_{flt}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"Saved {fname}")
