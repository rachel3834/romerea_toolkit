import sys
sys.path.append("/data01/aschweitzer/software")
import numpy as np
import pandas as pd
import os
from astropy.io import fits
from romerea_toolkit.hd5_utils import read_star_from_hd5_file
from romerea_toolkit import crossmatch

#paths
photo_dir = "/data01/aschweitzer/software/photo_copies"
output_dir = "/data01/aschweitzer/software/CV_Lightcurves/plots"
data_dir = "/data01/aschweitzer/data"
os.makedirs(output_dir, exist_ok=True)

#summarizing output
summary_file = os.path.join(output_dir, "CV_field_star_summary.txt")
with open(summary_file, "w") as f:
    f.write("field_id ra dec quadrant_id\n")

#thru rome field .fits files
for field_num in range(1, 21):
    print(f"\nProcessing field {field_num:02d}")
    
    crossmatch_file = os.path.join(data_dir, f"ROME/ROME-FIELD-{field_num:02d}/ROME-FIELD-{field_num:02d}_field_crossmatch.fits")
    match_file = os.path.join(data_dir, f"ROME-FIELD-{field_num:02d}_CV_matches.txt")
    
    if not os.path.exists(match_file):
        print(f"Missing CV match file for field {field_num}")
        continue

    if not os.path.exists(crossmatch_file):
        print(f"Missing crossmatch file for field {field_num}")
        continue

    match_df = pd.read_csv(match_file, delim_whitespace=True)
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(crossmatch_file, log=None)
    image_data = xmatch.images
    hjd = image_data["hjd"]
    filters = image_data["filter"]

    #get ra/dec/quad
    with fits.open(crossmatch_file) as hdul:
        field_index_data = hdul["FIELD_INDEX"].data
        ra_map = {row["field_id"]: row["ra"] for row in field_index_data}
        dec_map = {row["field_id"]: row["dec"] for row in field_index_data}
        quad_map = {row["field_id"]: row["quadrant"] for row in field_index_data}

    grouped = match_df.groupby("quadrant")
    for quadrant, group in grouped:
        phot_file = os.path.join(photo_dir, f"ROME-FIELD-{field_num:02d}_quad{quadrant}_photometry.hdf5")
        if not os.path.exists(phot_file):
            print(f"Missing photometry for field {field_num} quad {quadrant}")
            continue

        for _, row in group.iterrows():
            qid = int(row["quadrant_id"])
            field_id = int(row["field_id"])

            try:
                star_lc = read_star_from_hd5_file(phot_file, qid)
            except Exception as e:
                print(f"Failed reading quadrant {qid} from {phot_file}: {e}")
                continue

            norm_mag = star_lc[:, 7]
            norm_err = star_lc[:, 8]

            final_path = os.path.join(output_dir, f"field{field_id}_quad{quadrant}_qid{qid}_lc.txt")
            df = pd.DataFrame({
                "HJD": hjd,
                "Norm_Mag": norm_mag,
                "Norm_Mag_Err": norm_err,
                "Filter": filters
            })
            df.to_csv(final_path, index=False, sep=' ', float_format="%.6f")

            #add to summary file
            ra = ra_map.get(field_id, -99)
            dec = dec_map.get(field_id, -99)
            quad = quad_map.get(field_id, -1)

            with open(summary_file, "a") as f:
                f.write(f"{field_id} {ra:.6f} {dec:.6f} {quad}\n")

            print(f"Saved lightcurve and metadata for field {field_id}, qid {qid}")
