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

# filter data from images table
with fits.open(crossmatch_path) as hdul:
    filter_array = hdul["IMAGES"].data["filter"]  # (n_obs,)

#limiting to valid data only
min_total_obs = 140
valid_obs_per_star = np.sum((raw_data[:, :, QC_COL] == 0) & (raw_data[:, :, MAG_COL] > 0), axis=1)
valid_mask = valid_obs_per_star >= min_total_obs
valid_indices = np.where(valid_mask)[0]

# matching index master txt file
with open(os.path.join(output_dir, "valid_star_indices.txt"), "w") as f:
    for idx in valid_indices:
        f.write(f"{idx}\n")

# process PER Filter
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
        good = (star[:, QC_COL] == 0) & (star[:, MAG_COL] > 0)
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
    
    print(f"Saving scatterplot per filter for filter {flt} to {output_dir}!")

#now getting individual stars to make lcs
examples = {"const": {}, "var": {}}

for flt in filters:
    txt_path = os.path.join(output_dir, f"variability_std_mean_{flt}.txt")
    if not os.path.exists(txt_path):
        continue
    const_ids = []
    var_ids = []
    with open(txt_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["is_variable"] == "1":
                var_ids.append(int(row["star_index"]))
            else:
                const_ids.append(int(row["star_index"]))
    examples["const"][flt] = const_ids
    examples["var"][flt] = var_ids

# now finding common stars meeting reqs
filtered_examples = {"const": None, "var": None}

for kind in ["const", "var"]:
    commons = set.intersection(*[set(v) for v in examples[kind].values() if v])
    for idx in commons:
        all_filters_ok = True
        for flt in filters:
            filt_mask = filter_array == flt
            star = raw_data[idx, filt_mask, :]
            good = (star[:, QC_COL] == 0) & (star[:, MAG_COL] > 0)
            if np.sum(good) < 150:
                all_filters_ok = False
                break
        if all_filters_ok:
            filtered_examples[kind] = idx
            break

#plot GOOD example stars ONLY
for kind, idx in filtered_examples.items():
    if idx is None:
        print(f"No {kind} star found with ≥50 points in each filter.")
        continue
    for flt in filters:
        filt_mask = filter_array == flt
        star = raw_data[idx, filt_mask, :]
        good = (star[:, QC_COL] == 0) & (star[:, MAG_COL] > 0)
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

        print(f"Saved lightcurve for Star {idx} in filter {flt} to {output_dir}!")
