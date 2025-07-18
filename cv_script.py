import sys
sys.path.append("/data01/aschweitzer/software")

import numpy as np
import pandas as pd
import os
from astropy.io import fits
from romerea_toolkit.hd5_utils import read_star_from_hd5_file
from romerea_toolkit import crossmatch

photo_dir = "/data01/aschweitzer/software/photo_copies"
data_dir = "/data01/aschweitzer/data"
output_dir = "/data01/aschweitzer/software/CV_Lightcurves/plots"
microlia_out_dir = "/data01/aschweitzer/software/microlia_output/cv"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(microlia_out_dir, exist_ok=True)

summary_rows = []
all_lightcurve_rows = []
label_rows = []

# Loop through fields
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
    hjd = image_data["hjd"]
    filters_all = image_data["filter"]

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

            df_star = pd.DataFrame({
                "id": star_id,
                "time": hjd,
                "mag": star_lc[:, 7],
                "mag_err": star_lc[:, 8],
                "filter": filters_all.astype(str),
            })

            #drop any rows with invalid/missing data
            df_star = df_star.dropna()

            if df_star.empty:
                print(f"No valid observations for {star_id}, skipping.")
                continue

            # Save per-star lightcurve (optional)
            single_csv_path = os.path.join(output_dir, f"{star_id}.csv")
            df_star.to_csv(single_csv_path, index=False, float_format="%.6f")

            all_lightcurve_rows.append(df_star)
            label_rows.append({"id": star_id, "label": "CV"})
            summary_rows.append({
                "id": star_id,
                "field_id": field_id,
                "ra": ra,
                "dec": dec,
                "quadrant_id": quadrant_val,
                "n_obs": len(df_star),
            })

            print(f"Saved lightcurve for {star_id}")

#save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(output_dir, "microlia_CV_star_summary.csv"), index=False)

#save combined lightcurve file
combined_lc = pd.concat(all_lightcurve_rows, ignore_index=True)
combined_lc.to_csv(os.path.join(microlia_out_dir, "cv_microlia_lightcurves.csv"), index=False, float_format="%.6f")

#save labels
label_df = pd.DataFrame(label_rows)
label_df.to_csv(os.path.join(microlia_out_dir, "cv_microlia_labels.csv"), index=False)

print("\n Microlia data export complete!")
print(f"Lightcurves saved to {microlia_out_dir}/cv_microlia_lightcurves.csv")
print(f"Labels saved to {microlia_out_dir}/cv_microlia_labels.csv")
