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

#input table
df = pd.read_csv(input_csv)

#storage for microlia format
all_lc_rows = []
all_ids = set()

#read each row of table
for i, row in tqdm(df.iterrows(), total=len(df)):
    fits_path = row.get("fits", "")

    #skip if not a valid FITS path
    if not isinstance(fits_path, str) or not fits_path.endswith(".fits"):
        continue

    #convert relative path to full URL
    if fits_path.startswith("./"):
        fits_path = fits_path[2:]  # remove './'
    
    #making full url
    base_url = "https://exoplanetarchive.ipac.caltech.edu/data/ROME/data_reduction"
    full_url = f"{base_url}/{fits_path}"

    #making a local filename
    filename = os.path.basename(fits_path)
    local_path = os.path.join(output_dir, filename)

    #download FITS file if not already present
    if not os.path.exists(local_path):
        try:
            temp_path = download_file(full_url, cache=True, show_progress=False, timeout=30)
            os.rename(temp_path, local_path)
        except Exception as e:
            print(f"Failed to download {full_url}: {e}")
            continue

    #read downloaded FITS file
    try:
        with fits.open(local_path) as hdul:
            data = hdul[1].data
            hjd = data["hjd"]
            mag = data["mag"]
            magerr = data["magerr"]
            band = data["band"]

            obj_id = os.path.splitext(filename)[0]  #use FITS filename as ID
            all_ids.add(obj_id)

            for t, m, e, b in zip(hjd, mag, magerr, band):
                all_lc_rows.append({
                    "ID": obj_id,
                    "HJD": float(t),
                    "MAG": float(m),
                    "MAGERR": float(e),
                    "BAND": b.strip() if isinstance(b, str) else b.decode("utf-8").strip()
                })
    except Exception as e:
        print(f"Failed to process {local_path}: {e}")
        continue

#save MicroLIA format
lightcurve_df = pd.DataFrame(all_lc_rows)
lightcurve_df = lightcurve_df[["ID", "HJD", "MAG", "MAGERR", "BAND"]]  #enforcing specific order
lightcurve_df.to_csv(os.path.join(output_dir, "lightcurve.csv"), index=False)

label_df = pd.DataFrame({
    "ID": sorted(all_ids),
    "LABEL": [class_name] * len(all_ids)
})
label_df.to_csv(os.path.join(output_dir, "label.csv"), index=False)

print(f"\n Sfaved {len(all_ids)} labeled objects to: {output_dir}")
