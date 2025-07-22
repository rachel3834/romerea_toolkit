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
microlia_out_base = "/data01/aschweitzer/software/microlia_output"
TRAINING_BASE = os.path.join(microlia_out_base, "training_data")

# Create base output dir
os.makedirs(microlia_out_base, exist_ok=True)

# ROMEREA filter → Microlia filter mapping
filter_map = {"ip": "i", "rp": "r", "gp": "g"}

# Label for this dataset
label = "cv"

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
            print(f"Field id of star being read is {field_id}")

            try:
                star_lc = read_star_from_hd5_file(phot_file, qid)
                if not star_lc.dtype.isnative:
                    star_lc = star_lc.byteswap().newbyteorder()
            except Exception as e:
                print(f"Failed reading qid {qid} from {phot_file}: {e}")
                continue

            ra = ra_map.get(field_id, np.nan)
            dec = dec_map.get(field_id, np.nan)
            quadrant_val = quad_map.get(field_id, -1)

            star_id = f"field{field_id}_quad{quadrant}_qid{qid}"

            # LC columns
            mag = star_lc[:, 7]
            mag_err = star_lc[:, 8]
            valid = np.isfinite(mag) & np.isfinite(mag_err)

            if not np.any(valid):
                print(f"No valid observations for {star_id}, skipping.")
                continue

            filters_valid = [filter_map.get(f, f) for f in filters_all[valid]]
            df_full = pd.DataFrame({
                "time": hjd_all[valid],
                "mag": mag[valid],
                "mag_err": mag_err[valid],
                "filter": filters_valid
            })

            print(f"\n--- Processing star {star_id} ---")
            print(f"RA: {ra}, Dec: {dec}, Quadrant: {quadrant_val}")
            print(f"Total observations: {len(mag)}")
            print(f"Valid observations: {np.sum(valid)}")

            if df_full.empty:
                print(f"No valid data in DataFrame for {star_id}, skipping.")
                continue
            else:
                print(f"Total grouped filters: {df_full['filter'].nunique()}")
                print("Filter distribution:")
                print(df_full['filter'].value_counts())

            for filt, df_filt in df_full.groupby("filter"):
                print(f"\n-- Saving filter: {filt} --")
                out_dir = os.path.join(f"{TRAINING_BASE}_{filt}", label)
                os.makedirs(out_dir, exist_ok=True)

                filename = f"{star_id}.dat"
                filepath = os.path.join(out_dir, filename)

                df_to_save = df_filt[["time", "mag", "mag_err"]]
                print(f"Saving {len(df_to_save)} observations to {filepath}")
                df_to_save.to_csv(filepath, index=False, header=False, sep=" ", float_format="%.6f")

print("\nDone!")
