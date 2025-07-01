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
MAG_ERR_COL = 8
QC_COL = 16

with h5py.File(hdf5_path, "r") as f:
    dset = f["dataset_photometry"]
    raw_data = dset[:]

n_stars, n_obs, _ = raw_data.shape

# filter data from images table
with fits.open(crossmatch_path) as hdul:
    filter_array = hdul["IMAGES"].data["filter"]  # (n_obs,)
    field_ids_array = hdul["FIELD_INDEX"].data["field_id"]

#per-filter min
filters = ["rp", "gp", "ip"]
min_measurements = {"ip": 150, "gp": 100, "rp": 100}
filter_masks = {flt: (filter_array == flt) for flt in filters}

#now naming and applying filter-specific measurement #, magnitude, and qc_flag validity
per_filter_valid = {flt: [] for flt in filters}

for flt in filters:
    filt_mask = filter_masks[flt]
    for i in range(n_stars):
        star = raw_data[i, filt_mask, :]
        good = (star[:, QC_COL] == 0) & (star[:, MAG_COL] > 0)
        per_filter_valid[flt].append(np.sum(good) >= min_measurements[flt])

#now combining masks so we only get stars that meet EVERY filter criteria simultaneously
combined_valid_mask = np.logical_and.reduce([
    np.array(per_filter_valid[flt]) for flt in filters
])

valid_indices = np.where(combined_valid_mask)[0]

#overall index for star matching
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
    field_ids = []
    n_images = []

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
            field_ids.append(field_ids_array[i])
            n_images.append(np.sum(good))

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
    with open(txt_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["star_index", "mean_mag", "std_mag", "is_variable", "field_id", "n_images"])
        for idx, m, s, v, fid, nimg in zip(indices, means, stds, var_mask, field_ids, n_images):
            writer.writerow([idx, f"{m:.6f}", f"{s:.6f}", int(v), fid, nimg])
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
    plt.scatter(x[~is_var_sorted], y[~is_var_sorted], s=5, label="Constant", alpha=0.3)
    plt.scatter(x[is_var_sorted], y[is_var_sorted], s=5, color="orange", label="Variable", alpha=0.3)
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
        reader = csv.DictReader(f, delimiter=" ")
        for row in reader:
            if row["is_variable"] == "1":
                var_ids.append(int(row["star_index"]))
            else:
                const_ids.append(int(row["star_index"]))
    examples["const"][flt] = const_ids
    examples["var"][flt] = var_ids


#now finding common stars (making sure they're non-empty lists)
const_sets = [set(v) for v in examples["const"].values() if isinstance(v, list) and v]
const_common = set.intersection(*const_sets) if const_sets else set()

var_sets = [set(v) for v in examples["var"].values() if isinstance(v, list) and v]
var_common = set.intersection(*var_sets) if var_sets else set()


example_stars = {
    "const": next(iter(const_common)) if const_common else None,
    "var": next(iter(var_common)) if var_common else None
}

#plot GOOD example stars ONLY
for kind, idx in example_stars.items():
    if idx is None:
        print(f"No {kind} star found with enough points in each filter.")
        continue
    for flt in filters:
        filt_mask = filter_array == flt
        star = raw_data[idx, filt_mask, :]
        good = (star[:, QC_COL] == 0) & (star[:, MAG_COL] > 0)
        if np.sum(good) < min_measurements[flt]:
            print(f"Skipping star {idx} in filter {flt}: not enough points.")
            continue

        hjd = star[good, HJD_COL]
        mag = star[good, MAG_COL]
        errs = star[good, MAG_ERR_COL]
        photometry = star[good]


        sort = np.argsort(photometry[:, HJD_COL])
        hjd = hjd[sort]
        mag = mag[sort]
        errs = errs[sort]
        photometry = photometry[sort]

        #saving all photometry columns for chosen stars in txt file
        full_phot_file = os.path.join(output_dir, f"{kind}_star_{idx}_filter_{flt}_photometry_cols.txt")
        header = "HJD Inst_Mag Inst_Mag_Err Calib_Mag Calib_Mag_Err Corr_Mag Corr_Mag_Err Norm_Mag Norm_Mag_Err Phot_Scale Phot_Scale_Err Stamp_Idx Sky_Bkgd Sky_Bkgd_Err Residual_X Residual_Y QC_Flag Field_ID"
        photometry_with_field = np.column_stack([photometry, np.full((photometry.shape[0], 1), field_ids_array[idx])])
        
        print(f"photometry shape: {photometry.shape}")
        try:
            np.savetxt(full_phot_file, photometry_with_field, fmt="%.6f", header=header, delimiter="\t")
            print(f"Saved star {idx} photometry data to a .txt file in {output_dir}!")
        except Exception as e:
            print(f"Failed to save txt: {e}")

        print(f"{kind.title()} star {idx}, filter {flt}:")
        print(f"Mag error stats — min: {np.nanmin(errs)}, max: {np.nanmax(errs)}, NaNs: {np.isnan(errs).sum()}")

        plt.figure()
        plt.errorbar(hjd, mag, yerr=errs, fmt='o', markersize=3, alpha=0.7)
        plt.gca().invert_yaxis()
        plt.xlabel("HJD")
        plt.ylabel("Normalized Mag")
        plt.title(f"{kind.title()} Star {idx} - Filter {flt}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{kind}_star{idx}_filter_{flt}.png"))
        plt.close()

        print(f"Saved lightcurve for Star {idx} in filter {flt} to {output_dir}!")
