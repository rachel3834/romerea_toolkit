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
min_obs = {"rp": 40, "gp": 40, "ip": 40}
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
with open(rms_file, "r") as f:
    next(f)
    for line in f:
        star_idx, mean_mag, wmeans, werrors, rms, fit_rms, field_id, n_obs = line.split()
        if abs(float(rms) - float(fit_rms)) <= var_thresh:
            const_ids.append((int(star_idx), int(field_id)))

np.random.seed(42)
if len(const_ids) > 700:
    const_ids = random.sample(const_ids, 700)

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

    #now processing & saving star data for valid stars only
    for flt in filters:
        fm = filter_masks[flt]
        arr = data[star_idx, fm, :]
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0)

        hjd = arr[mask, HJD_COL]
        mag = arr[mask, MAG_COL]
        err = arr[mask, MAG_ERR_COL]
        med = np.median(mag)
        mean = np.mean(mag)

        sort = np.argsort(hjd)
        hjd, mag, err = hjd[sort], mag[sort], err[sort]

        #plotting
        plt.figure(figsize=(6, 4))
        plt.errorbar(hjd, mag, yerr=err, fmt='o', ms=3, alpha=0.3, color=filter_colors[flt])
        plt.axhline(med, color='red', linestyle='--', label='Median')
        plt.axhline(mean, color='green', linestyle=':', label='Mean')
        plt.gca().invert_yaxis()
        plt.xlabel("HJD")
        plt.ylabel("Mag")
        plt.title(f"Field {field_id} — Const Star {star_idx} in {flt}")
        plt.legend()
        plt.tight_layout()
        plot_name = f"field{field_id}_const_star{star_idx}_{flt}_lc.png"
        plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()

        #photometry txt save
        photometry = arr[mask]
        photometry_with_field = np.column_stack([photometry, np.full(len(photometry), field_id)])
        header = (
            "HJD Inst_Mag Inst_Mag_Err Calib_Mag Calib_Mag_Err Corr_Mag Corr_Mag_Err "
            "Norm_Mag Norm_Mag_Err Phot_Scale Phot_Scale_Err Stamp_Idx Sky_Bkgd "
            "Sky_Bkgd_Err Residual_X Residual_Y QC_Flag Field_ID"
        )
        txt_name = f"field{field_id}_const_star{star_idx}_{flt}_photometry.txt"
        np.savetxt(os.path.join(output_dir, txt_name), photometry_with_field, fmt="%.6f", header=header, delimiter="\t")
