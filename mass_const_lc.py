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
ogle_vars = "/data01/aschweitzer/software/CV_Lightcurves/Const_fits/ogle_var_ids/table_ROMESimplest.csv"
rome_table = pd.read_csv(ogle_vars)
os.makedirs(output_dir, exist_ok=True)

#starting values
var_thresh = 0.5
filters = ["rp", "gp", "ip"]
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

for j, flt in enumerate(filters):
    fm = filter_masks[flt]  #filter-specific mask on the time axis
    arr = data[:, fm, :] 
    
    #valid mask per observation:
    valid_mask = (arr[:, :, QC_COL] == 0) & (arr[:, :, MAG_COL] > 0) & (arr[:, :, MAG_ERR_COL] < 0.5)

    #counting valid obs per star in a filter
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
#limiting via ogle list of vars (exclude known variables)
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

        # get median uncertainties in each filter
med_unc = {}
for flt in filters:
    fm = filter_masks[flt]
    arr = data[star_idx, fm, :]
    mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0)
    med_unc[flt] = np.median(arr[mask, MAG_ERR_COL]) if np.sum(mask) > 0 else np.inf

    # exclude if median uncertainty > RMS in any filter
    if all(rms_dicts[flt].get(star_idx, np.inf) < med_unc[flt] for flt in filters):
        continue  # skip star if in all filters median uncertainty > RMS


    # also apply rms similarity threshold
    if (
        abs(rms_gp - rms_rp) > 0.5 or
        abs(rms_gp - rms_ip) > 0.5 or
        abs(rms_rp - rms_ip) > 0.5
    ):
        continue

    if abs(float(rms) - float(fit_rms)) <= var_thresh and star_idx in valid_set and star_idx not in exclude_star_indices:
        const_ids.append((star_idx, field_id, mean))

#now getting a random sample by binning by .5's
from collections import defaultdict

#collecting stars with mean mags in bins of 0.5
mag_bin_dict = defaultdict(list)
bin_width = 0.5

for star_idx, field_id, mean_mag in const_ids:
    if 13.5 <= mean_mag <= 21.0:
        bin_key = round(mean_mag / bin_width) * bin_width
        mag_bin_dict[bin_key].append((star_idx, field_id))

#now want to randomly sample from each bin
total_desired = 700
all_binned_ids = []

#how many stars per bin?
n_bins = len(mag_bin_dict)
per_bin = max(1, total_desired // n_bins)

np.random.seed(42)
for bin_key, star_list in mag_bin_dict.items():
    n_sample = min(per_bin, len(star_list))
    sampled = random.sample(star_list, n_sample)
    all_binned_ids.extend(sampled)


#cut down to EXACT total desired (if needed)
if len(all_binned_ids) > total_desired:
    all_binned_ids = random.sample(all_binned_ids, total_desired)


print(f"Selected {len(all_binned_ids)} constant stars!")

const_ids = all_binned_ids


for star_idx, field_id in const_ids:
    #now proceed to skip a star if it has invalid data in any filter
    skip = False
    for flt in filters:
        fm = filter_masks[flt]
        arr = data[star_idx, fm, :]
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0) & (arr[:, MAG_ERR_COL] < 0.5)
        if np.sum(mask) == 0:
            skip = True
            break
    if skip:
        continue  #skip this star entirely
   
   
   
   #now to get all filter data....
    all_filter_data = []

    for flt in filters:
        fm = filter_masks[flt]
        arr = data[star_idx, fm, :]
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0) & (arr[:, MAG_ERR_COL] < 0.5)

        if np.sum(mask) == 0:
            continue  #already checked for full data earlier

        hjd = arr[mask, HJD_COL]
        mag = arr[mask, MAG_COL]
        err = arr[mask, MAG_ERR_COL]

        #sorting by HJD
        sort = np.argsort(hjd)
        hjd, mag, err = hjd[sort], mag[sort], err[sort]

        all_filter_data.append((flt, hjd, mag, err))

        #save photometry per each star per filter
        photometry = arr[mask]
        field_id = int(field_id)
        photometry_with_field = np.column_stack([photometry, np.full(len(photometry), field_id)])
        header = (
            "HJD Inst_Mag Inst_Mag_Err Calib_Mag Calib_Mag_Err Corr_Mag Corr_Mag_Err "
            "Norm_Mag Norm_Mag_Err Phot_Scale Phot_Scale_Err Stamp_Idx Sky_Bkgd "
            "Sky_Bkgd_Err Residual_X Residual_Y QC_Flag Field_ID"
        )
        print(f"Saving txt file in {flt} for star idx {star_idx} field id {field_id}!")
        txt_name = f"field{field_id}_const_star{star_idx}_{flt}_photometry.txt"
        np.savetxt(os.path.join(output_dir, txt_name), photometry_with_field, fmt="%.6f", header=header, delimiter="\t")

    #now plotting 
    plt.figure(figsize=(7, 5))
    for flt, hjd, mag, err in all_filter_data:
        plt.errorbar(hjd, mag, yerr=err, fmt='o', ms=3, alpha=0.4, color=filter_colors[flt], label=flt)
        mean_mag = np.mean(mag)
        plt.axhline(mean_mag, color=filter_colors[flt], linestyle='--',
                    linewidth=1.2, alpha=0.6,
                    label=f"{flt} mean = {mean_mag:.2f}")
    plt.gca().invert_yaxis()
    plt.xlabel("HJD")
    plt.ylabel("Magnitude")
    plt.title(f"Field {field_id} — Const Star {star_idx} (All Filters)")
    plt.legend()
    plt.tight_layout()
    plot_name = f"field{field_id}_const_star{star_idx}_ALL_lc.png"
    print(f"Plotted star {star_idx} field id {field_id}")
    plt.savefig(os.path.join(output_dir, plot_name))
    plt.close()
