import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from astropy.io import fits
from romerea_toolkit import crossmatch, hd5_utils
import random

#paths
hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
crossmatch_path = "/data01/aschweitzer/data/ROME/ROME-FIELD-20/ROME-FIELD-20_field_crossmatch.fits"
output_dir = "CV_Lightcurves/Const_fits"
os.makedirs(output_dir, exist_ok=True)

#starting values
var_thresh = 0.5
filters = ["rp", "gp", "ip"]
min_obs = {"rp": 100, "gp": 100, "ip": 100}
filter_colors = {"rp": "red", "gp": "green", "ip": "blue"}

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

#minimum # obs per filter
valid = np.ones(n_stars, dtype=bool)
for flt in filters:
    fm = filter_masks[flt]
    counts = [(data[i, fm, QC_COL] == 0).sum() for i in range(n_stars)]
    valid &= np.array(counts) >= min_obs[flt]
valid_idx = np.where(valid)[0]

#getting constants from rms made in scatterplots_n_lcs.py
rms_file = os.path.join(output_dir, "variability_rms_rp.txt")
if not os.path.exists(rms_file):
    raise FileNotFoundError(f"{rms_file} not found... Try running scatterplots_n_lcs.py first!")

const_ids = []
valid_set = set(valid_idx.tolist())

with open(rms_file, "r") as f:
    next(f)
    for line in f:
        star_idx, mean_mag, wmeans, werrors, rms, fit_rms, field_id, n_obs = line.split()
        star_idx = int(star_idx)
        if abs(float(rms) - float(fit_rms)) <= var_thresh and star_idx in valid_set:
            const_ids.append((star_idx, field_id))

#now getting a random sample by binning by .5's
from collections import defaultdict

#collecting stars with mean mags in bins of 0.5
mag_bin_dict = defaultdict(list)
bin_width = 0.5

with open(rms_file, "r") as f:
    next(f)
    for line in f:
        star_idx, mean_mag, wmeans, werrors, rms, fit_rms, field_id, n_obs = line.split()
        star_idx = int(star_idx)
        field_id = int(field_id)
        mean_mag = float(mean_mag)
        if abs(float(rms) - float(fit_rms)) <= var_thresh and star_idx in valid_set:
            bin_key = round(mean_mag / bin_width)* bin_width
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
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0)
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
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0)

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
        photometry_with_field = np.column_stack([photometry, np.full(len(photometry), field_id)])
        header = (
            "HJD Inst_Mag Inst_Mag_Err Calib_Mag Calib_Mag_Err Corr_Mag Corr_Mag_Err "
            "Norm_Mag Norm_Mag_Err Phot_Scale Phot_Scale_Err Stamp_Idx Sky_Bkgd "
            "Sky_Bkgd_Err Residual_X Residual_Y QC_Flag Field_ID"
        )
        txt_name = f"field{field_id}_const_star{star_idx}_{flt}_photometry.txt"
        np.savetxt(os.path.join(output_dir, txt_name), photometry_with_field, fmt="%.6f", header=header, delimiter="\t")

    #now plotting 
    plt.figure(figsize=(7, 5))
    for flt, hjd, mag, err in all_filter_data:
        plt.errorbar(hjd, mag, yerr=err, fmt='o', ms=3, alpha=0.4, color=filter_colors[flt], label=flt)

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
