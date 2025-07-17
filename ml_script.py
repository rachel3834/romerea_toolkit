import os
import pandas as pd
import requests
from astropy.io import fits
from tqdm import tqdm

# paths and constants
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/workspace/TMP_Z62BKO_8520/ROME/tab1/data/data_reduction"
OUTPUT_DIR = "/data01/aschweitzer/software/microlia_output/ml"
LC_CSV = os.path.join(OUTPUT_DIR, "microlia_lightcurves.csv")
LABEL_CSV = os.path.join(OUTPUT_DIR, "microlia_labels.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load input CSV
input_csv = "/data01/aschweitzer/software/CV_Lightcurves/ml_lcs/ml_table.csv"
input_df = pd.read_csv(input_csv)

#filters and HDU names from inspecting fits file
filter_hdu_map = {
    "g": "LIGHTCURVE_SDSS_G",
    "r": "LIGHTCURVE_SDSS_R",
    "i": "LIGHTCURVE_SDSS_I"
}

#output storage
lightcurve_rows = []
labels = []

#now group input by star name (unique ID)
for star_name, star_group in tqdm(input_df.groupby("name"), desc="Processing stars"):

    fits_path = star_group["lc_file_path"].iloc[0]
    local_filename = os.path.join(OUTPUT_DIR, os.path.basename(fits_path))

    #save label if available
    if "label" in star_group.columns:
        label = star_group["label"].iloc[0]
        if pd.notna(label):
            labels.append({"id": star_name, "label": label})

    #download FITS if missing
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

    #open FITS and extract lightcurves
    try:
        with fits.open(local_filename) as hdul:
            for filt, hdu_name in filter_hdu_map.items():
                if hdu_name not in hdul:
                    print(f"Warning: {hdu_name} not found in {local_filename}, skipping filter {filt}")
                    continue

                data = hdul[hdu_name].data

                #good quality points filter
                good = (data["qc_flag"] == 0)
                if not good.any():
                    print(f"No good quality data for star {star_name}, filter {filt}")
                    continue

                hjd = data["HJD"][good]
                norm_mag = data["norm_mag"][good]
                norm_mag_err = data["norm_mag_error"][good]

                # Match rows in CSV for this filter
                mag_col = f"norm_mag_{filt}"
                mag_err_col = f"norm_mag_error_{filt}"
                if mag_col not in star_group.columns or mag_err_col not in star_group.columns:
                    print(f"Missing columns {mag_col} or {mag_err_col} for star {star_name}, skipping filter {filt}")
                    continue

                # Get only rows for this filter
                #mag/error columns existing per filter
                star_filter_data = star_group[[mag_col, mag_err_col]].reset_index(drop=True)

                # Length check (CSV vs FITS good data)
                if len(hjd) != len(star_filter_data):
                    print(f"Length mismatch for star {star_name} filter {filt}: FITS={len(hjd)} vs CSV={len(star_filter_data)}. Skipping filter.")
                    continue

                df = pd.DataFrame({
                    "id": star_name,
                    "field": star_group["field"].iloc[0],
                    "field_id": star_group["field_id"].iloc[0],
                    "time": hjd,
                    "mag": star_filter_data[mag_col],
                    "mag_err": star_filter_data[mag_err_col],
                    "filter": filt
                })
                lightcurve_rows.append(df)

    except Exception as e:
        print(f"Error reading FITS {local_filename}: {e}")
        continue

# Save combined lightcurve CSV
if lightcurve_rows:
    lc_df = pd.concat(lightcurve_rows, ignore_index=True)
    lc_df.to_csv(LC_CSV, index=False)
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
