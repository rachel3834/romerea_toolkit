import os
import pandas as pd
import requests
from astropy.io import fits
from tqdm import tqdm

# Constants and paths
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/workspace/TMP_Z62BKO_8520/ROME/tab1/data/data_reduction"
OUTPUT_DIR = "/data01/aschweitzer/software/microlia_output/ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LC_CSV = os.path.join(OUTPUT_DIR, "microlia_lightcurves.csv")
LABEL_CSV = os.path.join(OUTPUT_DIR, "microlia_labels.csv")

input_csv = "/data01/aschweitzer/software/CV_Lightcurves/ml_lcs/ml_table.csv"
input_df = pd.read_csv(input_csv)

# Map filters to FITS HDU names
filter_hdu_map = {
    "g": "LIGHTCURVE_SDSS_G",
    "r": "LIGHTCURVE_SDSS_R",
    "i": "LIGHTCURVE_SDSS_I"
}

lightcurve_rows = []
labels = []

for star_name, star_group in tqdm(input_df.groupby("name"), desc="Processing stars"):
    fits_path = star_group["lc_file_path"].iloc[0]
    local_filename = os.path.join(OUTPUT_DIR, os.path.basename(fits_path))

    # Save label if exists
    if "label" in star_group.columns:
        label = star_group["label"].iloc[0]
        if pd.notna(label):
            labels.append({"id": star_name, "label": label})

    # Download FITS if needed
    if not os.path.exists(local_filename):
        url = f"{BASE_URL}/{fits_path}"
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                f.write(r.content)
            print(f"Downloaded {local_filename}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            continue

    # Read FITS and extract filtered lightcurve data
    try:
        with fits.open(local_filename) as hdul:
            for filt, hdu_name in filter_hdu_map.items():
                if hdu_name not in hdul:
                    print(f"Warning: {hdu_name} missing in {local_filename}, skipping {filt}")
                    continue
                data = hdul[hdu_name].data
                good = (data["qc_flag"] == 0)
                if not good.any():
                    print(f"No good data for star {star_name} filter {filt}")
                    continue

                hjd = data["HJD"][good]
                norm_mag = data["norm_mag"][good]
                norm_mag_err = data["norm_mag_error"][good]

                df = pd.DataFrame({
                    "id": star_name,
                    "field": star_group["field"].iloc[0],
                    "field_id": star_group["field_id"].iloc[0],
                    "time": hjd,
                    "mag": norm_mag,
                    "mag_err": norm_mag_err,
                    "filter": filt
                })
                lightcurve_rows.append(df)
    except Exception as e:
        print(f"Error reading FITS {local_filename}: {e}")
        continue

# Save output lightcurves CSV
if lightcurve_rows:
    all_lcs = pd.concat(lightcurve_rows, ignore_index=True)
    all_lcs.to_csv(LC_CSV, index=False)
    print(f"Saved lightcurves to {LC_CSV}")
else:
    print("No lightcurve data extracted.")

# Save labels CSV
if labels:
    label_df = pd.DataFrame(labels).drop_duplicates(subset="id")
    label_df.to_csv(LABEL_CSV, index=False)
    print(f"Saved labels to {LABEL_CSV}")
else:
    print("No labels to save.")
