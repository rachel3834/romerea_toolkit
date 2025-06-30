import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import csv
import random
from astropy.io import fits

hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
output_dir = "CV_Lightcurves/Const_fits"
crossmatch_path = "/data01/aschweitzer/data/ROME/ROME-FIELD-20/ROME-FIELD-20_field_crossmatch.fits"
os.makedirs(output_dir, exist_ok=True)

MAG_COL = 7
HJD_COL = 0
QC_COL = 16

with h5py.File(hdf5_path, "r") as f:
    dset = f["dataset_photometry"]
    raw_data = dset[:]

n_stars, n_obs, _ = raw_data.shape

# Load filter data from crossmatch
with fits.open(crossmatch_path) as hdul:
    filter_array = hdul["IMAGES"].data["filter"]  # (n_obs,)

# Step 1: Determine which stars have valid data
valid_mask = np.array([np.any((raw_data[i, :, QC_COL] == 0) & (raw_data[i, :, MAG_COL] > -100)) for i in range(n_stars)])
valid_indices = np.where(valid_mask)[0]

# Step 2: Save master index list for matching across filters
with open(os.path.join(output_dir, "valid_star_indices.txt"), "w") as f:
    for idx in valid_indices:
        f.write(f"{idx}\n")

# Step 3: Process each filter with filter-specific QC and mag filtering
filters = ["rp", "gp", "ip"]

examples = {"const": {}, "var": {}}

for flt in filters:
    filt_mask = filter_array == flt

    means = []
    stds = []
    is_var = []
    indices = []

    for i in valid_indices:
        star = raw_data[i, filt_mask, :]
        good = (star[:, QC_COL] == 0) & (star[:, MAG_COL] > -100)
        mags = star[good, MAG_COL]
        if len(mags) > 0:
            mean_mag = np.mean(mags)
            std_mag = np.std(mags)
            means.append(mean_mag)
            stds.append(std_mag)
            indices.append(i)

    means = np.array(means)
    stds = np.array(stds)
    indices = np.array(indices)

    if len(means) < 10:
        print(f"Too few valid stars in filter {flt}, skipping...")
        continue

    fit_poly = np.polyfit(means, stds, 2)
    fit_fn = np.poly1d(fit_poly)
    fit_vals = fit_fn(means)
    upper_thresh = fit_vals + 0.3
    var_mask = stds > upper_thresh

    txt_path = os.path.join(output_dir, f"variability_std_mean_{flt}.txt")
    with open(txt_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["star_index", "mean_mag", "std_mag", "is_variable"])
        for idx, m, s, v in zip(indices, means, stds, var_mask):
            writer.writerow([idx, m, s, int(v)])
    print(f"Saved stats to {txt_path}")

    # Save for example LC matching
    examples["const"][flt] = indices[~var_mask].tolist()
    examples["var"][flt] = indices[var_mask].tolist()

    # Plotting
    sort_idx = np.argsort(means)
    x = means[sort_idx]
    y = stds[sort_idx]
    yfit = fit_fn(x)
    upper = yfit + 0.3
    lower = yfit - 0.3
    is_var_sorted = var_mask[sort_idx]

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(x[~is_var_sorted], y[~is_var_sorted], s=5, label="Constant", alpha=0.7)
    plt.scatter(x[is_var_sorted], y[is_var_sorted], s=5, color="orange", label="Variable", alpha=0.7)
    plt.plot(x, yfit, 'g-', label="Best-fit")
    plt.plot(x, upper, 'r--', label="±0.3 threshold")
    plt.plot(x, lower, 'r--')
    plt.xlabel("Mean Magnitude")
    plt.ylabel("Standard Deviation")
    plt.title(f"Field 20, Quad 4: Filter {flt}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"field20_quad4_{flt}_var_plot.png"))
    plt.close()

# Step 4: Find matching example stars across filters
const_common = set.intersection(*[set(v) for v in examples["const"].values() if isinstance(v, list)])
var_common = set.intersection(*[set(v) for v in examples["var"].values() if isinstance(v, list)])

example_stars = {
    "const": next(iter(const_common)) if const_common else None,
    "var": next(iter(var_common)) if var_common else None
}

for kind, idx in example_stars.items():
    if idx is None:
        continue
    for flt in filters:
        filt_mask = filter_array == flt
        star = raw_data[idx, filt_mask, :]
        good = (star[:, QC_COL] == 0) & (star[:, MAG_COL] > -100)
        hjd = star[good, HJD_COL]
        mag = star[good, MAG_COL]

        sort = np.argsort(hjd)
        hjd = hjd[sort]
        mag = mag[sort]

        plt.figure()
        plt.plot(hjd, mag, '.-')
        plt.gca().invert_yaxis()
        plt.xlabel("HJD")
        plt.ylabel("Normalized Mag")
        plt.title(f"{kind.title()} Star {idx} - Filter {flt}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{kind}_star{idx}_filter_{flt}.png"))
        plt.close()
