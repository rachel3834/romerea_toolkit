import os
import pandas as pd
import requests
from astropy.io import fits
from tqdm import tqdm

# Paths and constants
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/workspace/TMP_Z62BKO_8520/ROME/tab1/data/data_reduction"
OUTPUT_DIR = "/data01/aschweitzer/software/microlia_output/ml"
LC_CSV = os.path.join(OUTPUT_DIR, "microlia_lightcurves.csv")
LABEL_CSV = os.path.join(OUTPUT_DIR, "microlia_labels.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load input CSV
input_csv = "/data01/aschweitzer/software/CV_Lightcurves/ml_lcs/ml_table.csv"
input_df = pd.read_csv(input_csv)

# Detect filters from columns
mag_cols = [c for c in input_df.columns if c.startswith("norm_mag_")]
filters = [c.replace("norm_mag_", "") for c in mag_cols]

print(f"Detected filters: {filters}")

# Prepare output containers
lightcurve_rows = []
labels = []

# Iterate over each star by 'name'
for star_name, star_group in tqdm(input_df.groupby("name"), desc="Processing stars"):
    fits_path = star_group["lc_file_path"].iloc[0]

    # Save label if present
    if "label" in star_group.columns:
        label = star_group["label"].iloc[0]
        if pd.notna(label):
            labels.append({"id": star_name, "label": label})

    # Download FITS if missing
    local_filename = os.path.join(OUTPUT_DIR, os.path.basename(fits_path))
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

    # Open FITS once per star
    try:
        with fits.open(local_filename) as hdul:
            for filt in filters:
                table_name = f"table_{filt}"
                if table_name not in hdul:
                    print(f"Warning: {table_name} not found in {fits_path}, skipping filter {filt}")
                    continue

                hjd = hdul[table_name].data["Heliocentric Julian Date"]

                mag_col = f"norm_mag_{filt}"
                mag_err_col = f"norm_mag_err_{filt}"

                if mag_col not in star_group.columns or mag_err_col not in star_group.columns:
                    print(f"Missing mag or error column for filter {filt} for star {star_name}, skipping.")
                    continue

                # Sort by epoch if available, else keep order
                if "epoch" in star_group.columns:
                    star_filter_data = star_group.sort_values("epoch").reset_index(drop=True)
                else:
                    star_filter_data = star_group.reset_index(drop=True)

                # Length check
                if len(hjd) != len(star_filter_data):
                    print(f"Length mismatch for star {star_name} filter {filt}: FITS={len(hjd)} vs CSV={len(star_filter_data)}, skipping.")
                    continue

                # Build lightcurve dataframe
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
        print(f"Error opening FITS {local_filename}: {e}")
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
