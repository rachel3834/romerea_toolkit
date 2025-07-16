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
micrlout_dir = "/data01/aschweitzer/software/CV_Lightcurves/microlia_output/cv"
os.makedirs(output_dir, exist_ok=True)


summary_rows = []
all_lightcurve_rows = []
label_rows = []

#loop per field
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
    filters_all = image_data["filter"]

    #mapping field_id to ra, dec, quadrant
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

            #DataFrame of all observations for this star
            df_star = pd.DataFrame({
                "time": hjd,
                "filter": filters_all,
                "mag": star_lc[:, 7],      # norm_mag
                "mag_err": star_lc[:, 8],  # norm_mag_err
            })

            #wide format
            df_mag = df_star.pivot(index="time", columns="filter", values="mag")
            df_err = df_star.pivot(index="time", columns="filter", values="mag_err")

            #rename columns like'rp' -> 'mag_rp', 'mag_err_rp'
            df_mag.rename(columns=lambda c: f"mag_{c}", inplace=True)
            df_err.rename(columns=lambda c: f"mag_err_{c}", inplace=True)

            df_final = pd.concat([df_mag, df_err], axis=1).reset_index()

            #metadata columns added
            ra = ra_map.get(field_id, np.nan)
            dec = dec_map.get(field_id, np.nan)
            quadrant_val = quad_map.get(field_id, -1)

            df_final["field"] = f"ROME-FIELD-{field_num:02d}"
            df_final["field_id"] = field_id
            df_final["ra"] = ra
            df_final["dec"] = dec
            df_final["quadrant_id"] = quadrant_val

            #making sure there's a consistent column order (time, mag_rp, mag_err_rp, mag_gp, mag_err_gp, mag_ip, mag_err_ip, metadata) for microlia
            cols = ["time"]
            for f in ["rp", "gp", "ip"]:
                mag_col = f"mag_{f}"
                err_col = f"mag_err_{f}"
                if mag_col in df_final.columns:
                    cols.append(mag_col)
                if err_col in df_final.columns:
                    cols.append(err_col)
            cols += ["field", "field_id", "ra", "dec", "quadrant_id"]
            df_final = df_final[cols]

            #save CSV per star
            csv_path = os.path.join(output_dir, f"microlia_field{field_id}_quad{quadrant}_qid{qid}.csv")
            df_final.to_csv(csv_path, index=False, float_format="%.6f")

            summary_rows.append({
                "field": f"ROME-FIELD-{field_num:02d}",
                "field_id": field_id,
                "ra": ra,
                "dec": dec,
                "quadrant_id": quadrant_val,
                "qid": qid,
                "n_obs": len(df_final),
            })

            print(f"Saved star CSV: {csv_path}")


            #unique ID per star
            star_id = f"field{field_id}_quad{quadrant}_qid{qid}"
            df_final.insert(0, "id", star_id)  #'id' as first column

            #store rows for combined lightcurve
            all_lightcurve_rows.append(df_final)

            #add label row
            label_rows.append({"id": star_id, "label": "CV"})


#save full summary CSV to CV_Lightcurves/plots
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(output_dir, "microlia_CV_star_summary.csv")
summary_df.to_csv(summary_csv_path, index=False, float_format="%.6f")
print(f"\nSaved summary CSV: {summary_csv_path}")


#now combine and save all lightcurves into one Microlia CSV
combined_df = pd.concat(all_lightcurve_rows, ignore_index=True)
combined_lc_path = os.path.join(micrlout_dir, "cv_microlia_lightcurves.csv")
combined_df.to_csv(combined_lc_path, index=False, float_format="%.6f")

#save label file for Microlia as CSV
label_df = pd.DataFrame(label_rows)
label_path = os.path.join(micrlout_dir, "cv_microlia_labels.csv")
label_df.to_csv(label_path, index=False)

print(f"\nCombined lightcurves saved to: {combined_lc_path}")
print(f"Labels saved to: {label_path}")
