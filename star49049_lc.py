import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from astropy.io import fits
import random
from romerea_toolkit import crossmatch,hd5_utils

#paths
hdf5_path = "/data01/aschweitzer/software/photo_copies/ROME-FIELD-20_quad4_photometry.hdf5"
output_dir = "CV_Lightcurves/star49049_lcs"
crossmatch_path = "/data01/aschweitzer/data/ROME/ROME-FIELD-20/ROME-FIELD-20_field_crossmatch.fits"

os.makedirs(output_dir, exist_ok=True)


#listing star hjd (x-axis) range
base_hjd = 2_458_000.0

def get_hjd_range_for_filter(flt):
    if flt == "gp":
        # gp is offset relative to base_hjd
        return (-200, 100)
    else:
        # ip and rp use absolute HJD
        return (2.4578e6, 2.4581e6)
    



#column numbers
HJD_COL = 0
MAG_COL = 7
MAG_ERR_COL = 8
QC_COL = 16
QUAD_ID = 4

filters = ["rp", "gp", "ip"]
filter_colors = {"rp": "red", "gp": "green", "ip": "blue"}

#getting photometry from path now
with h5py.File(hdf5_path, "r") as f:
    data = f["dataset_photometry"][:]  # shape (n_stars, n_obs, n_cols)

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

filter_masks = {flt: im_filters == flt for flt in filters}

#find field_id = 49049
target_field_id = 49049

#field_id → star_idx
field_id_to_star_idx = {
    row["field_id"]: row["quadrant_id"]
    for row in field_idx
    if row["quadrant"] == QUAD_ID
}


if target_field_id not in field_id_to_star_idx:
    raise ValueError(f"FIELD ID {target_field_id} not found in quadrant {QUAD_ID}!")
star_idx = field_id_to_star_idx[target_field_id]

#now getting xy coords so i can insepct on ds9
star_entries = images[images["index"] == star_idx]

if len(star_entries) > 0:
    print(f"\n--- x-y coordinates for Star Index #{star_idx} (FIELD ID {target_field_id}) ---")
    for entry in star_entries:
        flt = entry["filter"]
        x = entry["sigma_x"]
        y = entry["sigma_y"]
        print(f"Filter {flt}: sigma_x = {x:.2f}, sigma_y = {y:.2f}")
else:
    print(f"No image data found for star_idx{star_idx}")



#now plotting all filters in one lightcurve
plt.figure(figsize=(8, 5))
for flt in filters:
    mask = filter_masks[flt]
    star_data = data[star_idx, mask, :]


    
    #filtering out for quality
    valid = (star_data[:, QC_COL] == 0) & (star_data[:, MAG_COL] > 0)
    hjd = star_data[valid, HJD_COL]
    mag = star_data[valid, MAG_COL]
    err = star_data[valid, MAG_ERR_COL]
    
    if flt == "gp":
        hjd += base_hjd  # shift gp back again

    # HJD range mask
    flt_range = get_hjd_range_for_filter(flt)
    hjd_mask = (hjd >= flt_range[0]) & (hjd <= flt_range[1])
    hjd, mag, err = hjd[hjd_mask], mag[hjd_mask], err[hjd_mask]

    if len(hjd) == 0:
        continue

    plt.errorbar(hjd, mag, yerr=err, fmt='o', ms=3, label=flt, alpha=0.7, color=filter_colors[flt])

plt.gca().invert_yaxis()
plt.xlabel("HJD")
plt.ylabel("Magnitude")
plt.title(f"Field ID {target_field_id} — Lightcurve (All filters)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"field{target_field_id}_combined_lc.png"))
plt.close()
print(f"Saved combined lightcurve for star {target_field_id}.")

#now for individual plots in each filter (same structure as above but per star)
for flt in filters:
    mask = filter_masks[flt]
    star_data = data[star_idx, mask, :]
    
    valid = (star_data[:, QC_COL] == 0) & (star_data[:, MAG_COL] > 0)
    hjd = star_data[valid, HJD_COL]
    mag = star_data[valid, MAG_COL]
    err = star_data[valid, MAG_ERR_COL]

    if flt == "gp":
        hjd += base_hjd  # shift gp back again

    flt_range = get_hjd_range_for_filter(flt)
    hjd_mask = (hjd >= flt_range[0]) & (hjd <= flt_range[1])

    hjd, mag, err = hjd[hjd_mask], mag[hjd_mask], err[hjd_mask]

    if len(hjd) == 0:
        print(f"No data for {flt} in HJD range.")
        continue

    plt.figure(figsize=(6, 4))
    plt.errorbar(hjd, mag, yerr=err, fmt='o', ms=3, alpha=0.6, color=filter_colors[flt])
    plt.gca().invert_yaxis()
    plt.xlabel("HJD")
    plt.ylabel("Magnitude")
    plt.title(f"Star {target_field_id} — {flt} filter")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"star{target_field_id}_{flt}_lc.png"))
    plt.close()
    print(f"Saved {flt} filter lightcurve for star {target_field_id}.")





