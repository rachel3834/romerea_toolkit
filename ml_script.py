import pandas as pd
from astropy.io import fits
from astropy.utils.data import download_file
import os

#paths
class_name = "ml" 
base_dir = "/data01/aschweitzer/software/microlia_output"
input_csv = os.path.join(base_dir, class_name, f"{class_name}_table.csv")
output_dir = os.path.join(base_dir, class_name)
os.makedirs(output_dir, exist_ok=True)

#data storage
lightcurve_rows = []
label_rows = []

df = pd.read_csv(input_csv)

for i, row in df.iterrows():
    name = str(row.get("name") or row.get("Name")).strip()
    field = str(row.get("field") or row.get("Field")).strip()
    star_id = f"{field}_{name}"

    #last column is the FITS URL
    fits_url = row.iloc[-1]
    if not isinstance(fits_url, str) or not fits_url.endswith(".fits"):
        print(f"Skipping {star_id}: Invalid FITS URL")
        continue

    try:
        #download fits to cache so we can actually access the lc data we need inside
        fits_path = download_file(fits_url, cache=True, timeout=30)

        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            if data is None or len(data) == 0:
                print(f" Empty data: {star_id}")
                continue

            if not all(k in data.columns.names for k in ["HJD", "MAG_CALIB", "MAG_CALIB_ERR"]):
                print(f"Missing columns for {star_id}")
                continue

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

            label_rows.append({
                "id": star_id,
                "label": class_name.upper()
            })

            print(f"Downloaded/processed {star_id} ({len(hjd)} points)")

    except Exception as e:
        print(f"Error processing {star_id}: {e}")
        continue

#now saving as lc and label csv's that are req. for microlia array format
lc_path = os.path.join(output_dir, f"{class_name}_microlia_lightcurves.csv")
label_path = os.path.join(output_dir, f"{class_name}_microlia_labels.csv")

pd.DataFrame(lightcurve_rows).to_csv(lc_path, index=False)
pd.DataFrame(label_rows).to_csv(label_path, index=False)

print(f"\n Saved lightcurves to {lc_path}")
print(f" Sdaved labels to {label_path}")
