import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import csv
from astropy.io import fits
import random
from romerea_toolkit import crossmatch,hd5_utils

#setting my paths
hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
crossmatch_path = "/data01/aschweitzer/data/ROME/ROME-FIELD-20/ROME-FIELD-20_field_crossmatch.fits"
output_dir = "CV_Lightcurves/Const_fits/twice_threshold"
os.makedirs(output_dir, exist_ok=True)

var_thresh = 0.5

xmatch = crossmatch.CrossMatchTable()
xmatch.load(crossmatch_path, log=None)

phot_data = hd5_utils.read_phot_from_hd5_file(hdf5_path)

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
            err = arr[mask, MAG_ERR_COL]
            if np.any(err <= 0):
                continue  # skip bad errors so nan doesn't occur with err_sq_inv
            
            med_mag = np.median(mags)
            err_sq_inv = 1.0/(err * err)
            wmean = (mags * err_sq_inv).sum()/(err_sq_inv.sum())
            werror = np.sqrt(1.0/(err_sq_inv.sum()))
            dmags = mags - wmean #residuals

            rms = np.sqrt((dmags**2 * err_sq_inv).sum()/(err_sq_inv.sum()))
          

            mean_mag = mags.mean()
            std_mag = mags.std()
            stars.append((i, mean_mag, wmean, werror, rms, mapping.get(i,-1), mask.sum()))

    #skip stars index < 10
    if len(stars)<10:
        print(f"Skip {flt}: too few stars...")
        continue
    stars = np.array(stars, dtype=object)
    idxs, means, wmeans, werrors, rms_vals, fields, counts = zip(*stars)
    means, rms_vals = np.array(means), np.array(rms_vals)

    

    #making a fit
    mask_finite = np.isfinite(means) & np.isfinite(rms_vals)
    means_clean = means[mask_finite]
    rms_vals_clean = rms_vals[mask_finite]

    fit = np.poly1d(np.polyfit(means_clean, rms_vals_clean, 2))
    fit_rms = fit(means)

    #writing txt files per filter
    out = sorted(zip(idxs, means, wmeans, werrors, rms_vals, fit_rms, fields, counts), key=lambda x:x[0])
    with open(os.path.join(output_dir,f"variability_rms_{flt}.txt"),"w") as f:
        f.write("star_idx mean_mag wmean werror RMS fit_rms field_id n_obs\n")
        for i,m,wm,we,r,fr,fid,nobs in out:
            f.write(f"{i} {m:.4f} {wm:.4f} {we:.4f} {r:.4f} {fr:.4f} {fid} {nobs}\n")

  

    #now plotting the scatterplot using rms best-fit
    x = np.array([o[1] for o in out])
    y = np.array([o[4] for o in out])
    yfit = np.array([o[5] for o in out])

    
    is_variable = np.abs(y - yfit) > var_thresh

    #check
    print(f"x values are {x}")
    print(f"y values are {y}")
    print(f"y values are {yfit}")

    residuals = y - yfit

    if np.any(np.isnan(residuals)):
        print("nans found in residuals!")
        print("y:", y)
        print("yfit:", yfit)
        print("residuals:", residuals)
    else:
        plt.hist(residuals, bins=50)
        plt.axvline(0, color="k", linestyle="--")
        plt.axvline(var_thresh, color="red", linestyle="--", label="Threshold")
        plt.axvline(-var_thresh, color="red", linestyle="--")
        plt.xlabel("RMS - Fit_RMS")
        plt.title(f"Residuals: {flt}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"field20_quad4_{flt}_residuals_hist.png"))


    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    yfit_sorted = yfit[sort_idx]

    

    plt.figure(figsize=(8,6), dpi=300)
    plt.scatter(x[~is_variable], y[~is_variable], alpha=0.3, s=5, color="steelblue", label="Constant")
    plt.scatter(x[is_variable], y[is_variable], alpha=0.3, s=5, color="red", label="Variable")
    plt.plot(x_sorted, yfit_sorted, 'g-', label="Best-fit RMS")
    plt.xlabel("Mean Mag"); plt.ylabel("RMS")
    plt.title(f"Field20 Quad4 — RMS vs Mag ({flt})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"field20_quad4_{flt}_rms.png"))
    plt.close()
    print(f"Saved RMS scatterplot for {flt}!")


#now making individual lc plots/txt files for var and const stars


#loading one filter txt as a base to pick const/var stars from
rms_file = os.path.join(output_dir, "variability_rms_rp.txt")
const_ids = []
var_ids = []

#now reading this file
with open(rms_file, "r") as f:
    next(f)
    for line in f:
        star_idx, mean_mag, wmeans, werrors, rms, fit_rms, field_id, n_obs = line.split()
        star_idx = int(star_idx)
        rms = float(rms)
        fit_rms = float(fit_rms)

        if abs(rms - fit_rms) <= var_thresh:
            const_ids.append(star_idx)
        else:
            var_ids.append(star_idx)

#randomly picking three stars!
random.seed(42)
const_selected = random.sample(const_ids, min(3, len(const_ids)))
var_selected = random.sample(var_ids, min(3, len(var_ids)))

#print which stars selected of each type
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
                print(f"No good data for star {idx} in {flt}...")
                continue
            
            #making vars for plot
            hjd = arr[mask, HJD_COL]
            mag = arr[mask, MAG_COL]
            err = arr[mask, MAG_ERR_COL]
            med = np.median(mag)
            mean = np.mean(mag)

            #sorting
            sort = np.argsort(hjd)
            hjd, mag, err = hjd[sort], mag[sort], err[sort]
            
            #build plot and save
            plt.figure(figsize=(6, 4))
            plt.errorbar(hjd, mag, yerr=err, fmt='o', ms=3, alpha=0.3)
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
            print(f"Saved LC: {savepath}!")

            #now saving photometry (all 16 cols + field_id) for each star in txt
            phot_file = os.path.join(output_dir, f"{kind}_star{idx}_{flt}_photometry.txt")
            filt_obs_indices = np.where(fm)[0]                  # obs indices for this filter
            good_obs_indices = filt_obs_indices[mask]           # valid obs indices (QC==0, mag>0)
            photometry = arr[mask]                              # shape: (n_good_obs, 17)

            #adding field_id label
            field_id = mapping.get(idx, -1)
            photometry_with_field = np.column_stack([photometry, np.full(len(photometry), field_id)])

            header = ("HJD Inst_Mag Inst_Mag_Err Calib_Mag Calib_Mag_Err Corr_Mag Corr_Mag_Err "
                    "Norm_Mag Norm_Mag_Err Phot_Scale Phot_Scale_Err Stamp_Idx Sky_Bkgd "
                    "Sky_Bkgd_Err Residual_X Residual_Y QC_Flag Field_ID")

            #save
            np.savetxt(phot_file, photometry_with_field, fmt="%.6f", header=header, delimiter="\t")
            print(f"Saved photometry data for star {idx} in {flt} to {phot_file}!")
