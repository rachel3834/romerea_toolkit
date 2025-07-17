import os
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
from tqdm import tqdm

#config
class_name = "ml"
dir_name = "ml_lcs"
base_dir = "/data01/aschweitzer/software/CV_Lightcurves"
input_csv = os.path.join(base_dir, dir_name, f"{class_name}_table.csv")
final_dir = "/data01/aschweitzer/software/microlia_output"
output_dir = os.path.join(final_dir, class_name)
os.makedirs(output_dir, exist_ok=True)

#load input table
df = pd.read_csv(input_csv)

#output containers for microlia
all_lc_rows = []
all_ids = set()

#url base
base_url = "https://exoplanetarchive.ipac.caltech.edu/workspace/TMP_Z62BKO_8520/ROME/tab1/data/data_reduction"

#process each fits file
for _, row in tqdm(df.iterrows(), total=len(df)):
    fits_path = row.get("fits", "")

    #skip if no valid fits filename
    if not isinstance(fits_path, str) or not fits_path.endswith(".fits"):
        print(f"Skipping (invalid fits path): {fits_path}")
        continue

    filename = os.path.basename(fits_path)
    obj_id = os.path.splitext(filename)[0]
    local_path = os.path.join(output_dir, filename)

    full_url = f"{base_url}/{fits_path.strip('./')}"

    #download file only if missing
    if not os.path.exists(local_path):
        try:
            temp_path = download_file(full_url, cache=True, show_progress=False, timeout=30)
            os.rename(temp_path, local_path)
        except Exception as e:
            print(f"Failed to download {full_url}: {e}")
            continue

    #now try opening and parsing fits file (careful with 'id')
    try:
        with fits.open(local_path) as hdul:
            for band_key in ["table_i", "table_g", "table_r"]:
                if band_key not in hdul:
                    continue
                table = hdul[band_key].data
                hjd = table["Heliocentric Julian Date"]
                norm_mag = table["Normalized mag"]
                norm_mag_err = table["Normalized mag (error)"]

                for t, m, e in zip(hjd, norm_mag, norm_mag_err):
                    all_lc_rows.append({
                        "id": obj_id,
                        "time": float(t),
                        "mag": float(m),
                        "mag_err": float(e),
                        "filter": band_key[-1]  #looking at last char: 'i', 'g', or 'r'
                    })
                all_ids.add(obj_id)
    except Exception as e:
        print(f"Failed to process {local_path}: {e}")
        continue

#save lightcurve CSV (for microlia)
lightcurve_df = pd.DataFrame(all_lc_rows)
lightcurve_df = lightcurve_df[["id", "time", "mag", "mag_err", "filter"]]
lightcurve_df.to_csv(os.path.join(output_dir, "lightcurve.csv"), index=False)

#save label CSV (for microlia)
label_df = pd.DataFrame({
    "id": sorted(all_ids),
    "label": [class_name] * len(all_ids)
})
label_df.to_csv(os.path.join(output_dir, "label.csv"), index=False)

print(f"\nSaved {len(all_ids)} labeled objects to: {output_dir}")
