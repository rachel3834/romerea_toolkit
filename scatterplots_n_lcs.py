import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import csv
import random

hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
output_dir = "CV_Lightcurves/Const_fits"
os.makedirs(output_dir, exist_ok=True)

MAG_COL = 7
HJD_COL = 0
QC_COL = 16

with h5py.File(hdf5_path, "r") as f:
    dset = f["dataset_photometry"]
    raw_data = dset[:]  #expected actual shape is (n_stars, n_obs, n_cols)

n_stars = raw_data.shape[0]

#now limiting the dataset. if there is invalid data in one filter, removing that star's data in ALL filters
valid_mask = np.array([np.any((raw_data[i, :, QC_COL] == 0) & (raw_data[i, :, MAG_COL] > 0)) for i in range(n_stars)])
valid_indices = np.where(valid_mask)[0]

#now, making an overall txt so we can create matching indices across filters
with open(os.path.join(output_dir, "valid_star_indices.txt"), "w") as f:
    for idx in valid_indices:
        f.write(f"{idx}\n")

#now processing statistics PER filter using valid_indices so that star ids are consistent across filters
filters = ["rp", "gp", "ip"]

for flt in filters:
    means = []
    stds = []
    is_var = []
    indices = []

    for i in valid_indices:
        star = raw_data[i]
        good = (star[:, QC_COL] == 0) & (star[:, MAG_COL] > 0)
        mags = star[good, MAG_COL]
        if len(mags) > 0:
            mean_mag = np.mean(mags)
            std_mag = np.std(mags)
            means.append(mean_mag)
            stds.append(std_mag)
            indices.append(i)

    #updating stats
    means = np.array(means)
    stds = np.array(stds)
    indices = np.array(indices)

    #notify us if there aren't enough stars in the filter (unlikely, low limit)
    if len(means) < 10:
        print(f"Too few valid stars in filter: {flt}, skipping!")
        continue

    #scatterplot fits
    fit_poly = np.polyfit(means, stds, 2)
    fit_fn = np.poly1d(fit_poly)
    fit_vals = fit_fn(means)
    upper_thresh = fit_vals + 0.3
    var_mask = stds > upper_thresh

    #saving .txt per filter
    txt_path = os.path.join(output_dir, f"variability_std_mean_{flt}.txt")
    with open(txt_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["star_index", "mean_mag", "std_mag", "is_variable"])
        for idx, m, s, v in zip(indices, means, stds, var_mask):
            writer.writerow([idx, m, s, int(v)])
    print(f"Saved stats to {txt_path}")

    #now plotting per filter
    sort_idx = np.argsort(means)
    x = means[sort_idx]
    y = stds[sort_idx]
    yfit = fit_fn(x)
    upper = yfit + 0.3
    lower = yfit - 0.3
    is_var_sorted = var_mask[sort_idx]

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(x[~is_var_sorted], y[~is_var_sorted], s=5, color="steelblue", label="Constant", alpha=0.7)
    plt.scatter(x[is_var_sorted], y[is_var_sorted], s=5, color="orange", label="Variable", alpha=0.7)
    plt.plot(x, yfit, 'g-', label="Best fit")
    plt.plot(x, upper, 'r--', label="+-/0.3 threshold")
    plt.plot(x, lower, 'r--')
    plt.xlabel("Mean Magnitude")
    plt.ylabel("Standard Deviation")
    plt.title(f"Field 20 Q4, Filter: {flt}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"field20_quad4_{flt}_scatterplot.png"))
    plt.close()

#now doing INDIVIDUAL lightcurves per filter using ONE var and ONE const. star
examples = {}
for flt in filters:
    txt_path = os.path.join(output_dir, f"variability_std_mean_{flt}.txt")
    if not os.path.exists(txt_path):
        continue
    with open(txt_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        const_ids = []
        var_ids = []
        for row in reader:
            if row["is_variable"] == "1":
                var_ids.append(int(row["star_index"]))
            else:
                const_ids.append(int(row["star_index"]))
        if const_ids:
            examples.setdefault("const", []).append(random.choice(const_ids))
        if var_ids:
            examples.setdefault("var", []).append(random.choice(var_ids))

#making sure it's the SAME STAR across filters

#since examples["const"] is a list of integers, need to turn each integer into a set before set.intersection
const_lists = examples.get('const', [])
var_lists = examples.get('var', [])

const_common = set(const_lists) if const_lists else set()
var_common = set(var_lists) if var_lists else set()

example_stars = {
    "const": next(iter(const_common)) if const_common else None,
    "var": next(iter(var_common)) if var_common else None
}

for kind, idx in example_stars.items():
    if idx is None:
        continue
    for flt in filters:
        star = raw_data[idx]
        good = (star[:, QC_COL] == 0) & (star[:, MAG_COL] > 0)
        hjd = star[good, HJD_COL]
        mag = star[good, MAG_COL]

        #now actually plotting lightcurves
        plt.figure()
        plt.plot(hjd, mag, '.-')
        plt.gca().invert_yaxis()
        plt.xlabel("HJD")
        plt.ylabel("Normalized Mag")
        plt.title(f"{kind.title()} Star {idx}, Filter: {flt}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{kind}_star{idx}_filter_{flt}.png"))
        plt.close()

