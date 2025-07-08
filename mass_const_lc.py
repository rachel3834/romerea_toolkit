import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from astropy.io import fits
from romerea_toolkit import crossmatch, hd5_utils

#copied from scatterplots script (setup)
hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
crossmatch_path = "/data01/aschweitzer/data/ROME/ROME-FIELD-20/ROME-FIELD-20_field_crossmatch.fits"
rms_file = "CV_Lightcurves/Const_fits/variability_rms_rp.txt"
output_dir = "CV_Lightcurves/all_const_lcs"
os.makedirs(output_dir, exist_ok=True)

QC_COL = 16
MAG_COL = 7
MAG_ERR_COL = 8
HJD_COL = 0
QUAD_ID = 4
filters = ["rp", "gp", "ip"]
var_thresh = 0.5

with h5py.File(hdf5_path, "r") as f:
    data = f["dataset_photometry"][:]

with fits.open(crossmatch_path) as hdul:
    im_filters = hdul["IMAGES"].data["filter"]
    field_idx = hdul["FIELD_INDEX"].data
    filter_masks = {flt: im_filters == flt for flt in filters}

    #now match star_idx → field_id
    mapping = {
        row["quadrant_id"]: row["field_id"]
        for row in field_idx
        if row["quadrant"] == QUAD_ID
    }

#load in possible constant stars from variability_rms_rp.txt
const_ids = []
with open(rms_file, "r") as f:
    next(f)
    for line in f:
        star_idx, _, _, _, rms, fit_rms, field_id, _ = line.split()
        if abs(float(rms) - float(fit_rms)) <= var_thresh:
            const_ids.append((int(star_idx), int(field_id)))

#select constants
const_ids = const_ids[:700]
print(f"Selected {len(const_ids)} constants!")

#loop per star: make plot and txt
for star_idx, field_id in const_ids:
    for flt in filters:
        fm = filter_masks[flt]
        arr = data[star_idx, fm, :]
        mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0)
        if np.sum(mask) == 0:
            print(f"No valid data for star {star_idx} in {flt}")
            continue

        hjd = arr[mask, HJD_COL]
        mag = arr[mask, MAG_COL]
        err = arr[mask, MAG_ERR_COL]
        med = np.median(mag)
        mean = np.mean(mag)

        #sort by hjd
        sort = np.argsort(hjd)
        hjd, mag, err = hjd[sort], mag[sort], err[sort]

        #plot the lc
        plt.figure(figsize=(6, 4))
        plt.errorbar(hjd, mag, yerr=err, fmt='o', ms=3, alpha=0.3)
        plt.axhline(med, color='red', linestyle='--', label='Median')
        plt.axhline(mean, color='green', linestyle=':', label='Mean')
        plt.gca().invert_yaxis()
        plt.xlabel("HJD")
        plt.ylabel("Mag")
        plt.title(f"Const Star {star_idx} — {flt} — Field {field_id}")
        plt.legend()
        plt.tight_layout()
        fname = f"const_star{star_idx}_field{field_id}_{flt}_lc.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"Saved LC: {fname}")

        #now saving photometry file as txt in format given by romerea_toolkit github
        phot_file = os.path.join(output_dir, f"const_star{star_idx}_field{field_id}_{flt}_photometry.txt")
        photometry_with_field = np.column_stack([arr[mask], np.full(np.sum(mask), field_id)])
        header = ("HJD Inst_Mag Inst_Mag_Err Calib_Mag Calib_Mag_Err Corr_Mag Corr_Mag_Err "
                  "Norm_Mag Norm_Mag_Err Phot_Scale Phot_Scale_Err Stamp_Idx Sky_Bkgd "
                  "Sky_Bkgd_Err Residual_X Residual_Y QC_Flag Field_ID")
        np.savetxt(phot_file, photometry_with_field, fmt="%.6f", header=header, delimiter="\t")
        print(f"Saved photometry: {phot_file}")
