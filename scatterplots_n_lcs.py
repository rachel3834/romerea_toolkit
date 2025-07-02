import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import csv
from astropy.io import fits
import random

#setting my paths
hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
crossmatch_path = "/data01/aschweitzer/data/ROME/ROME-FIELD-20/ROME-FIELD-20_field_crossmatch.fits"
output_dir = "CV_Lightcurves/Const_fits"
os.makedirs(output_dir, exist_ok=True)

#defining column indices
MAG_COL = 7
HJD_COL = 0
MAG_ERR_COL = 8
QC_COL = 16
QUAD_ID = 4  # quadrant we're processing

#now loading photometry in
with h5py.File(hdf5_path, "r") as f:
    data = f["dataset_photometry"][:]  # shape expected: (n_stars, n_obs, n_cols)
n_stars, _, _ = data.shape

#now loading crossmatch tables
with fits.open(crossmatch_path) as hdul:
    images = hdul["IMAGES"].data
    im_filters = images["filter"]  # shape: (n_obs,)
    field_idx = hdul["FIELD_INDEX"].data
    #map from (quadrant_id index in HDF5) → field_id index in field_index (crossmatch.fits)
    mapping = {
        row["quadrant_id"]: row["field_id"]
        for row in field_idx
        if row["quadrant"] == QUAD_ID
    }

#setting filter configs
filters = ["rp","gp","ip"]
min_obs = {"rp":100,"gp":100,"ip":150}
filter_masks = {flt: im_filters == flt for flt in filters}

#limiting dataset to stars with at least min_obs in each filter
valid = np.ones(n_stars, bool)
for flt in filters:
    fm = filter_masks[flt]
    counts = [(data[i,fm,QC_COL]==0).sum() for i in range(n_stars)]
    valid &= np.array(counts) >= min_obs[flt]
valid_idx = np.where(valid)[0]

#now making RMS plot per filter
for flt in filters:
    fm = filter_masks[flt]

    stars = []
    for i in valid_idx:
        arr = data[i,fm,:]
        mask = (arr[:,QC_COL]==0)&(arr[:,MAG_COL]>0)
        if mask.sum()>0:
            mags = arr[mask,MAG_COL]
            med_mag = np.median(mags)
            rms = np.sqrt(np.mean((mags-med_mag)**2))
            mean_mag = mags.mean()
            stars.append((i, mean_mag, rms, mapping.get(i,-1), mask.sum()))

    if len(stars)<10:
        print(f"Skip {flt}: too few stars")
        continue
    stars = np.array(stars, dtype=object)
    idxs, means, rms_vals, fields, counts = zip(*stars)
    means, rms_vals = np.array(means), np.array(rms_vals)

    fit = np.poly1d(np.polyfit(means, rms_vals, 2))
    fit_rms = fit(means)

    out = sorted(zip(means, rms_vals, fit_rms, idxs, fields, counts), key=lambda x:x[0])
    with open(os.path.join(output_dir,f"variability_rms_{flt}.txt"),"w") as f:
        f.write("star_idx mean_mag RMS fit_rms field_id n_obs\n")
        for m,r,fr,i,fid,nobs in out:
            f.write(f"{i} {m:.4f} {r:.4f} {fr:.4f} {fid} {nobs}\n")

    x = np.array([o[0] for o in out])
    y = np.array([o[1] for o in out])
    yfit = np.array([o[2] for o in out])
    plt.figure(figsize=(8,6), dpi=300)
    plt.scatter(x, y, alpha=0.3, s=5, label="Stars")
    plt.plot(x, yfit, 'g-', label="Best-fit RMS")
    plt.xlabel("Mean Magnitude"); plt.ylabel("RMS")
    plt.title(f"Field20 Quad4 — RMS vs Mag ({flt})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"field20_quad4_{flt}_rms.png"))
    plt.close()
    print(f"Saved RMS scatterplot for {flt}")


#now for part two:
#loading one filter txt as a base to pick const/var stars from
rms_file = os.path.join(output_dir, "variability_rms_rp.txt")
threshold_offset = 0.0001
const_ids = []
var_ids = []

#now reading this file
with open(rms_file, "r") as f:
    next(f)
    for line in f:
        star_idx, mean_mag, rms, fit_rms, field_id, n_obs = line.split()
        star_idx = int(star_idx)
        rms = float(rms)
        fit_rms = float(fit_rms)
        if rms <= fit_rms + threshold_offset:
            const_ids.append(star_idx)
        else:
            var_ids.append(star_idx)

#randomly picking three stars!
random.seed(42)
const_selected = random.sample(const_ids, 3)
var_selected = random.sample(var_ids, 3)

all_selected = {"const": const_selected, "var": var_selected}
print("Selected constant star indices:", const_selected)
print("Selected variable star indices:", var_selected)

#now plotting lcs
for kind, ids in all_selected.items():
    for idx in ids:
        for flt in filters:
            fm = filter_masks[flt]
            arr = data[idx, fm, :]
            mask = (arr[:, QC_COL] == 0) & (arr[:, MAG_COL] > 0)
            if np.sum(mask) == 0:
                print(f"No good data for star {idx} in {flt}")
                continue

            hjd = arr[mask, HJD_COL]
            mag = arr[mask, MAG_COL]
            err = arr[mask, MAG_ERR_COL]
            med = np.median(mag)
            mean = np.mean(mag)

            sort = np.argsort(hjd)
            hjd, mag, err = hjd[sort], mag[sort], err[sort]

            plt.figure(figsize=(6, 4))
            plt.errorbar(hjd, mag, yerr=err, fmt='o', ms=3, alpha=0.6)
            plt.axhline(med, color='red', linestyle='--', label='Median')
            plt.axhline(mean, color='green', linestyle=':', label='Mean')
            plt.gca().invert_yaxis()
            plt.xlabel("HJD")
            plt.ylabel("Mag")
            plt.title(f"{kind.title()} Star {idx} in {flt}")
            plt.legend()
            plt.tight_layout()
            savepath = os.path.join(output_dir, f"{kind}_star{idx}_{flt}_lc.png")
            plt.savefig(savepath)
            plt.close()
            print(f"Saved LC: {savepath}")