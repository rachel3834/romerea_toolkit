import pandas as pd
import os
from astropy.io import fits

#paths
input_csv = "/data01/aschweitzer/software/CV_Lightcurves/ml_lcs/ml_table.csv"  #table
output_dir = "/data01/aschweitzer/software/microlia_output/ml" 
label = "ML"  # or RRLYR, LPV, etc.

base_dir = "/data01/aschweitzer/data/ROME"
os.makedirs(output_dir, exist_ok=True)

#input table of stars
df = pd.read_csv(input_csv)
lightcurve_rows = []
label_rows = []

for _, row in df.iterrows():
    field_id = int(row["field_id"])
    field = row["field"].strip()
    name = row["name"].strip()
    star_id = str(field_id)  #unique ID for Microlia to use

    fits_path = os.path.join(base_dir, field, f"{name}.fits")
    if not os.path.exists(fits_path):
        print(f"Missing file: {fits_path}")
        continue

    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            hjd = data["HJD"]
            mag = data["MAG_CALIB"]
            mag_err = data["MAG_CALIB_ERR"]
            filters = data["FILTER"] if "FILTER" in data.columns.names else ["rp"] * len(hjd)

            for t, m, e, f in zip(hjd, mag, mag_err, filters):
                lightcurve_rows.append({
                    "id": star_id,
                    "time": t,
                    "mag": m,
                    "mag_err": e,
                    "filter": f
                })

            label_rows.append({"id": star_id, "label": label})

    except Exception as e:
        print(f"Error with {fits_path}: {e}")
        continue

#save microlia files (lc + label csvs)
lightcurve_df = pd.DataFrame(lightcurve_rows)
label_df = pd.DataFrame(label_rows)

lightcurve_path = os.path.join(output_dir, f"{label.lower()}_microlia_lightcurves.csv")
label_path = os.path.join(output_dir, f"{label.lower()}_microlia_labels.csv")

lightcurve_df.to_csv(lightcurve_path, index=False)
label_df.to_csv(label_path, index=False)

print(f"\n Lightcurves saved to: {lightcurve_path}")
print(f"Labels saved to: {label_path}")
