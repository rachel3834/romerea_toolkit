import sys
sys.path.append("/data01/aschweitzer/software")

import numpy as np
import pandas as pd
import os
from astropy.io import fits
from romerea_toolkit.hd5_utils import read_star_from_hd5_file
from romerea_toolkit import crossmatch
from tqdm import tqdm

photo_dir = "/data01/aschweitzer/software/photo_copies"
data_dir = "/data01/aschweitzer/data"
output_dir = "/data01/aschweitzer/software/CV_Lightcurves/plots"
microlia_out_base = "/data01/aschweitzer/software/microlia_output/cv"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(microlia_out_base, exist_ok=True)

#filter mapping from ROMEREA filters to Microlia filters
filter_map = {"ip": "i", "rp": "r", "gp": "g"}

#microlia training base dir root (per filter and label)
TRAINING_BASE = "/data01/aschweitzer/software/microlia_output/training_data"

summary_rows = []
all_lightcurve_rows = []


for field_num in range(1, 21):
    print(f"\nProcessing field {field_num:02d}")

    crossmatch_file = os.path.join(
        data_dir, f"ROME/ROME-FIELD-{field_num:02d}/ROME-FIELD-{field_num:02d}_field_crossmatch.fits")
    match_file = os.path.join(data_dir, f"ROME-FIELD-{field_num:02d}_CV_matches.txt")

    if not os.path.exists(match_file) or not os.path.exists(crossmatch_file):
        print(f"Missing data for field {field_num}")
        continue

    match_df = pd.read_csv(match_file, delim_whitespace=True)
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(crossmatch_file, log=None)
    image_data = xmatch.images
    hjd_all = image_data["hjd"]
    filters_all = image_data["filter"].astype(str)

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
                print(f"Failed reading qid {qid} from {phot_file}: {e}")
                continue

            ra = ra_map.get(field_id, np.nan)
            dec = dec_map.get(field_id, np.nan)
            quadrant_val = quad_map.get(field_id, -1)

            star_id = f"field{field_id}_quad{quadrant}_qid{qid}"

            #LC columns
            mag = star_lc[:, 7]
            mag_err = star_lc[:, 8]
            valid = np.isfinite(mag) & np.isfinite(mag_err)

            if not np.any(valid):
                print(f"No valid observations for {star_id}, skipping.")
                continue

            #fine filter for each observation and map it
            #filters_all matches time axis of star_lc for all stars — so now we can find filter per obs:
            filters_valid = [filter_map.get(f, f) for f in filters_all[valid]]

            # group data by filter
            df_full = pd.DataFrame({
                "time": hjd_all[valid],
                "mag": mag[valid],
                "mag_err": mag_err[valid],
                "filter": filters_valid
            })

            if df_full.empty:
                print(f"No valid data for {star_id}, skipping")
                continue

            #save csv files with time hjd, mag, mag_err
            for filt, df_filt in df_full.groupby("filter"):
                # output directory per filter and label
                out_dir = os.path.join(f"{TRAINING_BASE}_{filt}")
                os.makedirs(out_dir, exist_ok=True)

                filename = f"{star_id}.csv"
                filepath = os.path.join(out_dir, filename)

                df_to_save = df_filt[["time", "mag", "mag_err"]]
                df_to_save.to_csv(filepath, index=False, header=False, float_format="%.6f")

                print(f"Saved to {filepath}")

            #combined CSV analysis
            df_full["id"] = star_id
            all_lightcurve_rows.append(df_full)

            #summary info
            summary_rows.append({
                "id": star_id,
                "field_id": field_id,
                "ra": ra,
                "dec": dec,
                "quadrant_id": quadrant_val,
                "n_obs": len(df_full)
            })

#now save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(output_dir, "microlia_CV_star_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary CSV to {summary_path}")

#save combined CSV for analysis
if all_lightcurve_rows:
    combined_lc = pd.concat(all_lightcurve_rows, ignore_index=True)
    combined_path = os.path.join(microlia_out_base, "cv_microlia_lightcurves.csv")
    combined_lc.to_csv(combined_path, index=False, float_format="%.6f")
    print(f"Saved combined lightcurve CSV to {combined_path}")

print("\nDone!")
