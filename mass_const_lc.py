import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from astropy.io import fits
from romerea_toolkit import crossmatch, hd5_utils
import random
import pandas as pd

#paths
hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
crossmatch_path = "/data01/aschweitzer/data/ROME/ROME-FIELD-20/ROME-FIELD-20_field_crossmatch.fits"
output_dir = "CV_Lightcurves/Const_fits"
final_dir = "/data01/aschweitzer/software/microlia_output/const"
ogle_vars = "/data01/aschweitzer/software/CV_Lightcurves/Const_fits/ogle_var_ids/table_ROMESimplest.csv"
rome_table = pd.read_csv(ogle_vars)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

summary_path = os.path.join(output_dir, "const_star_summary.txt")
summary_header = "field_id star_id filter weighted_mean RMS median_unc n_obs\n"
with open(summary_path, "w") as f:
    f.write(summary_header)

#starting values
var_thresh = 0.25
filters = ["rp", "gp", "ip"]
filter_rename_map = {"rp": "r", "gp": "g", "ip": "i"}   #rename filters for microlia

min_obs = {"rp": 100, "gp": 100, "ip": 100}
filter_colors = {"rp": "red", "gp": "green", "ip": "blue"}
rms_dicts = {"rp": {}, "gp": {}, "ip": {}}

#columns
HJD_COL = 0
MAG_COL = 7
MAG_ERR_COL = 8
QC_COL = 16
QUAD_ID = 4

#now loading in xmatches
xmatch = crossmatch.CrossMatchTable()
xmatch.load(crossmatch_path, log=None)

with fits.open(crossmatch_path) as hdul:
    images = hdul["IMAGES"].data
    im_filters = images["filter"]
    field_idx = hdul["FIELD_INDEX"].data
    field_map = {row["quadrant_id"]: row["field_id"]
                 for row in field_idx if row["quadrant"] == QUAD_ID}
with h5py.File(hdf5_path, "r") as f:
    data = f["dataset_photometry"][:]
n_stars, _, _ = data.shape

#filter masks!
filter_masks = {flt: im_filters == flt for flt in filters}

#array for valid mask
obs_matrix = np.zeros((n_stars, len(filters)), dtype=int)

#-----------------------------#
#check effect of QC flag filtering
total_valid_obs = 0
total_raw_obs = 0

for j, flt in enumerate(filters):
    fm = filter_masks[flt]
    arr = data[:, fm, :]

    #with QC filtering
    qc_mask = (arr[:, :, QC_COL] == 0) & (arr[:, :, MAG_COL] > 0) & (arr[:, :, MAG_ERR_COL] < 0.5)
    total_valid_obs += qc_mask.sum()

    #without QC filtering
    raw_mask = (arr[:, :, MAG_COL] > 0) & (arr[:, :, MAG_ERR_COL] < 0.5)
    total_raw_obs += raw_mask.sum()

print(f"\n[Diagnostic] Total valid obs (QC=0):   {total_valid_obs}")
print(f"[Diagnostic] Total raw obs (ignoring QC): {total_raw_obs}")
print(f"[Diagnostic] Fraction kept after QC filtering: {total_valid_obs / total_raw_obs:.3f}\n")
#------------------------------#

#now actually building the lcs and txts valid masks
for j, flt in enumerate(filters):
    fm = filter_masks[flt]  #filter-specific mask on the time axis
    arr = data[:, fm, :]
    valid_mask = (arr[:, :, QC_COL] == 0) & (arr[:, :, MAG_COL] > 0) & (arr[:, :, MAG_ERR_COL] < 0.5)
    obs_matrix[:, j] = valid_mask.sum(axis=1)

#print counts
print("Star  #rp  #gp  #ip")
for i in range(n_stars):
    print(f"{i:5d} {obs_matrix[i,0]:4d} {obs_matrix[i,1]:4d} {obs_matrix[i,2]:4d}")

#bool masks for thresholds
valid_mask = (
    (obs_matrix[:, 0] >= min_obs["rp"]) &
    (obs_matrix[:, 1] >= min_obs["gp"]) &
    (obs_matrix[:, 2] >= min_obs["ip"])
)

valid_idx = np.where(valid_mask)[0]
valid_set = set(valid_idx.tolist())

#save matrix
np.savetxt(os.path.join(output_dir, "obs_counts_matrix.txt"),
           np.column_stack([np.arange(n_stars), obs_matrix]),
           fmt="%d", header="StarID rp gp ip", delimiter="\t")

#getting constants from rms made in scatterplots_n_lcs.py
rms_file = os.path.join(output_dir, "variability_rms_rp.txt")
if not os.path.exists(rms_file):
    raise FileNotFoundError(f"{rms_file} not found... Try running scatterplots_n_lcs.py first!")

const_ids = []
exclude_star_indices = set(rome_table["field_id"].values.astype(int))

#now getting dicts for gp and ip
for flt in filters:
    fm = filter_masks[flt]
    for i in range(n_stars):
        arr = data[i, fm, :]
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0) & (arr[:, MAG_ERR_COL] < 0.5)
        mags = arr[mask, MAG_COL]
        if len(mags) > 0:
            rms = np.std(mags)
            rms_dicts[flt][i] = rms

with open(rms_file, "r") as f:
    next(f)
    for line in f:
        star_idx, mean_mag, wmeans, werrors, rms, fit_rms, field_id, n_obs = line.split()
        star_idx = int(star_idx)
        mean = float(mean_mag)
        rms_gp = rms_dicts["gp"].get(star_idx)
        rms_ip = rms_dicts["ip"].get(star_idx)
        rms_rp = rms_dicts["rp"].get(star_idx)

        if None in (rms_gp, rms_ip, rms_rp):
            continue  # skip if any missing

        med_unc = {}
        for flt in filters:
            fm = filter_masks[flt]
            arr = data[star_idx, fm, :]
            mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0)
            med_unc[flt] = np.median(arr[mask, MAG_ERR_COL]) if np.sum(mask) > 0 else np.inf

        if any(med_unc[flt] > rms_dicts[flt].get(star_idx, np.inf) for flt in filters):
            print(f"Excluded star {star_idx} due to high uncertainty in one or more filters.")
            continue

        if (
            abs(rms_gp - rms_rp) > 0.5 or
            abs(rms_gp - rms_ip) > 0.5 or
            abs(rms_rp - rms_ip) > 0.5
        ):
            continue

        if abs(float(rms) - float(fit_rms)) <= var_thresh and star_idx in valid_set and star_idx not in exclude_star_indices:
            const_ids.append((star_idx, field_id, mean))

#bin stars by mean mag and sample
from collections import defaultdict

mag_bin_dict = defaultdict(list)
bin_width = 0.5

for star_idx, field_id, mean_mag in const_ids:
    if 13.5 <= mean_mag <= 21.0:
        bin_key = round(mean_mag / bin_width) * bin_width
        mag_bin_dict[bin_key].append((star_idx, field_id))

total_desired = 700
all_binned_ids = []

n_bins = len(mag_bin_dict)
per_bin = max(1, total_desired // n_bins)

np.random.seed(42)
for bin_key, star_list in mag_bin_dict.items():
    n_sample = min(per_bin, len(star_list))
    sampled = random.sample(star_list, n_sample)
    all_binned_ids.extend(sampled)

if len(all_binned_ids) > total_desired:
    all_binned_ids = random.sample(all_binned_ids, total_desired)

print(f"Selected {len(all_binned_ids)} constant stars!")

# Plotting and saving photometry files for const stars
for star_idx, field_id in all_binned_ids:
    plt.figure(figsize=(7, 5))
    summary_lines = []

    for offset, flt in enumerate(filters):
        arr = data[star_idx, filter_masks[flt], :]
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0) & (arr[:, MAG_ERR_COL] < 0.5)
        if np.sum(mask) == 0:
            continue
        hjd = arr[mask, HJD_COL]
        mag = arr[mask, MAG_COL]
        err = arr[mask, MAG_ERR_COL]
        weights = 1.0 / np.square(err)
        wmean = np.average(mag, weights=weights)
        residual = mag - wmean + 0.3 * offset

        rms = np.std(mag)
        med_u = np.median(err)
        n_obs = len(hjd)
        
        # Use renamed filter for summary and plotting labels
        renamed_flt = filter_rename_map.get(flt, flt)
        summary_lines.append(f"{field_id} {star_idx} {renamed_flt} {wmean:.4f} {rms:.4f} {med_u:.4f} {n_obs}")

        plt.errorbar(hjd, residual, yerr=err, fmt='o', ms=3, alpha=0.4,
                     color=filter_colors[flt],
                     label=f"{renamed_flt}: μ={wmean:.2f}, RMS={rms:.2f}, σₘ={med_u:.2f}, N={n_obs}")

        plt.axhline(0.3 * offset, color=filter_colors[flt], linestyle='--', linewidth=1.0, alpha=0.6)

        phot_with_fid = np.column_stack([arr[mask], np.full(n_obs, field_id)]).astype(np.float64)
        header = (
            f"# Star ID: {star_idx}, Field ID: {field_id}, Filter: {renamed_flt}\n"
            f"# Weighted Mean = {wmean:.4f}, RMS = {rms:.4f}, Median Uncertainty = {med_u:.4f}, N_obs = {n_obs}\n"
            "HJD Inst_Mag Inst_Mag_Err Calib_Mag Calib_Mag_Err Corr_Mag Corr_Mag_Err "
            "Norm_Mag Norm_Mag_Err Phot_Scale Phot_Scale_Err Stamp_Idx Sky_Bkgd "
            "Sky_Bkgd_Err Residual_X Residual_Y QC_Flag Field_ID"
        )

        np.savetxt(
            os.path.join(output_dir, f"field{field_id}_const_star{star_idx}_{renamed_flt}_photometry.txt"),
            phot_with_fid, fmt="%.6f", header=header, delimiter="\t"
        )

    for line in summary_lines:
        with open(summary_path, "a") as f:
            f.write(line + "\n")

    plt.axhline(0, color="gray", lw=0.5, linestyle="--")
    plt.gca().invert_yaxis()
    plt.xlabel("HJD")
    plt.ylabel("Residual Magnitude")
    plt.title(f"Residual Lightcurve — Field {field_id} Star {star_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"field{field_id}_const_star{star_idx}_ALL_resid_lc.png"))
    plt.close()

# FINAL CSV SAVE FILE FOR MICROLIA
csv_summary_path = os.path.join(output_dir, "const_star_photometry_summary.csv")
summary_rows = []

with fits.open(crossmatch_path) as hdul:
    field_index_table = hdul["FIELD_INDEX"].data

for star_idx, field_id in all_binned_ids:
    row_data = {
        "field": "ROME-FIELD-20",
        "field_id": int(field_id)
    }

    match = field_index_table[field_index_table["field_id"] == int(field_id)]
    if len(match) > 0:
        row_data["ra"] = match["ra"][0]
        row_data["dec"] = match["dec"][0]
        row_data["quadrant_id"] = match["quadrant_id"][0]
    else:
        row_data["ra"] = np.nan
        row_data["dec"] = np.nan
        row_data["quadrant_id"] = np.nan

    for flt in filters:
        arr = data[star_idx, filter_masks[flt], :]
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0) & (arr[:, MAG_ERR_COL] < 0.5)
        if np.sum(mask) == 0:
            row_data[f"{filter_rename_map[flt]}_norm_mag"] = np.nan
            row_data[f"{filter_rename_map[flt]}_norm_mag_err"] = np.nan
            continue

        mag = arr[mask, MAG_COL]
        err = arr[mask, MAG_ERR_COL]
        weights = 1.0 / np.square(err)
        wmean = np.average(mag, weights=weights)
        med_u = np.median(err)

        row_data[f"{filter_rename_map[flt]}_norm_mag"] = round(wmean, 4)
        row_data[f"{filter_rename_map[flt]}_norm_mag_err"] = round(med_u, 4)

    summary_rows.append(row_data)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(csv_summary_path, index=False)
print(f"\nSaved full star summary CSV to: {csv_summary_path}")

# lightcurve CSV for Microlia
lightcurve_rows = []

for star_idx, field_id in all_binned_ids:
    for flt in filters:
        arr = data[star_idx, filter_masks[flt], :]
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0) & (arr[:, MAG_ERR_COL] < 0.5)

        if np.sum(mask) == 0:
            continue

        hjd = arr[mask, HJD_COL]
        mag = arr[mask, MAG_COL]
        err = arr[mask, MAG_ERR_COL]

        renamed_flt = filter_rename_map.get(flt, flt)

        for t, m, e in zip(hjd, mag, err):
            lightcurve_rows.append({
                "id": f"{field_id}_{star_idx}",
                "time": t,
                "mag": m,
                "mag_err": e,
                "filter": renamed_flt
            })

lightcurve_df = pd.DataFrame(lightcurve_rows)
lightcurve_csv_path = os.path.join(final_dir, "const_microlia_lightcurves.csv")
lightcurve_df.to_csv(lightcurve_csv_path, index=False)
print(f"Microlia-compatible lightcurve CSV saved to: {lightcurve_csv_path}")

label_rows = []

for star_idx, field_id in all_binned_ids:
    label_rows.append({
        "id": f"{field_id}_{star_idx}",
        "label": "CONST"
    })

label_df = pd.DataFrame(label_rows)
label_csv_path = os.path.join(final_dir, "const_microlia_labels.csv")
label_df.to_csv(label_csv_path, index=False)
print(f"Microlia-compatible label CSV saved to: {label_csv_path}")
