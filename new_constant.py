import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pandas as pd

# === Paths ===
hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
crossmatch_path = "/data01/aschweitzer/data/ROME/ROME-FIELD-20/ROME-FIELD-20_field_crossmatch.fits"
output_dir = "CV_Lightcurves/Const_fits"
os.makedirs(output_dir, exist_ok=True)

# === Load photometry data ===
with h5py.File(hdf5_path, "r") as f:
    dset = f["dataset_photometry"]
    data = dset[:]  # shape (num_stars, num_obs)
    columns = dset.dtype.names

# Column indices
hjd_col = 0
mag_col = 1
err_col = 2
qid_col = 3
obj_id_col = 4
qc_flag_col = 16

# === Load quadrant_id → filter mapping ===
cross_df = pd.read_csv(crossmatch_path, delim_whitespace=True)
cross_df.columns = cross_df.columns.str.lower()
qid_to_filter = dict(zip(cross_df["quadrant_id"], cross_df["filter"]))

# Get filter name for each (star, obs) using quadrant_id
qid_matrix = data[:, :, qid_col]
filter_matrix = np.vectorize(lambda q: qid_to_filter.get(int(q), "unknown"))(qid_matrix)

# === Loop over filters ===
unique_filters = np.unique(filter_matrix)

for flt in unique_filters:
    if flt == "unknown":
        continue

    print(f"Processing filter: {flt}")

    # Only keep photometry in this filter and with QC == 0
    is_flt = filter_matrix == flt
    is_qc_good = data[:, :, qc_flag_col] == 0
    valid = is_flt & is_qc_good

    # Get magnitudes where valid
    mag_data = np.where(valid, data[:, :, mag_col], np.nan)
    means = np.nanmean(mag_data, axis=1)
    stds = np.nanstd(mag_data, axis=1)

    # Keep only stars with finite values
    valid_mask = ~np.isnan(means) & ~np.isnan(stds)
    if np.sum(valid_mask) < 10:
        print(f"Too few valid stars for filter {flt}, skipping...")
        continue

    valid_means = means[valid_mask]
    valid_stds = stds[valid_mask]

    # Fit polynomial curve
    fit_poly = np.polyfit(valid_means, valid_stds, deg=2)
    fit_fn = np.poly1d(fit_poly)
    fit_vals = fit_fn(valid_means)
    upper_thresh = fit_vals + 0.3
    lower_thresh = fit_vals - 0.3
    is_variable = valid_stds > upper_thresh

    # Sort for smooth plotting
    sort_idx = np.argsort(valid_means)
    x = valid_means[sort_idx]
    y = valid_stds[sort_idx]
    yfit = fit_fn(x)
    upper = yfit + 0.3
    lower = yfit - 0.3
    is_var_sorted = is_variable[sort_idx]

    # Plot
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(x[~is_var_sorted], y[~is_var_sorted], s=5, label="Constant", alpha=0.7)
    plt.scatter(x[is_var_sorted], y[is_var_sorted], s=5, color="orange", label="Variable", alpha=0.7)
    plt.plot(x, yfit, 'g-', label="Best-fit")
    plt.plot(x, upper, 'r--', label="±0.3 threshold")
    plt.plot(x, lower, 'r--')
    plt.xlabel("Mean Magnitude")
    plt.ylabel("Standard Deviation")
    plt.title(f"Field 20, Quad 4 — Filter: {flt}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"field20_quad4_{flt}_var_plot.png"))
    plt.close()
