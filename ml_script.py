import os
import pandas as pd
import requests
from astropy.io import fits
from tqdm import tqdm

# Constants and paths
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/workspace/TMP_Z62BKO_8520/ROME/tab1/data/data_reduction"
OUTPUT_DIR = "/data01/aschweitzer/software/microlia_output/ml"
TRAINING_DIR = "/data01/aschweitzer/software/microlia_output/training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)

LC_CSV = os.path.join(OUTPUT_DIR, "ml_microlia_lightcurves.csv")
LABEL_CSV = os.path.join(OUTPUT_DIR, "ml_microlia_labels.csv")

input_csv = "/data01/aschweitzer/software/CV_Lightcurves/ml_lcs/ml_table.csv"
input_df = pd.read_csv(input_csv)

#if no label column, add "ml"
if "label" not in input_df.columns:
    print("No 'label' column found in input CSV; adding default label='ml' for all stars.")
    input_df["label"] = "ml"
elif input_df["label"].isna().all():
    print("'label' column exists but all values are NaN; filling with default label='ml'.")
    input_df["label"] = "ml"

#map filters to FITS HDU names
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

    # Save label if exists and non-null
    label = star_group["label"].iloc[0]
    if pd.notna(label):
        labels.append({"id": star_name, "label": label})

    #download FITS if needed
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

    #read FITS and extract filtered lightcurve data
    try:
        with fits.open(local_filename) as hdul:
            star_dfs = []
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
                    "time": hjd,
                    "mag": norm_mag,
                    "mag_err": norm_mag_err,
                    "filter": filt
                })
                star_dfs.append(df)
                lightcurve_rows.append(df)  # for combined CSV saving

            #combine all filters for this star into one DataFrame
            if star_dfs:
                star_df = pd.concat(star_dfs, ignore_index=True)

                # sve to training data directory in Microlia format:
                star_dir = os.path.join(TRAINING_DIR, label)
                os.makedirs(star_dir, exist_ok=True)

                star_filename = f"star_{star_name}.csv"
                star_filepath = os.path.join(star_dir, star_filename)

                # Save with no header, no index
                star_df[["time", "mag", "mag_err", "filter"]].to_csv(
                    star_filepath, index=False, header=False, float_format="%.6f"
                )

                print(f"Saved Microlia training lightcurve for {star_name} in label folder {label}")

    except Exception as e:
        print(f"Error reading FITS {local_filename}: {e}")
        continue

#save combined CSV and labels CSV 
if lightcurve_rows:
    all_lcs = pd.concat(lightcurve_rows, ignore_index=True)
    all_lcs.to_csv(LC_CSV, index=False)
    print(f"Saved combined lightcurves CSV to {LC_CSV}")
else:
    print("No lightcurve data extracted.")

if labels:
    label_df = pd.DataFrame(labels).drop_duplicates(subset="id")
    label_df.to_csv(LABEL_CSV, index=False)
    print(f"Saved labels CSV to {LABEL_CSV}")
else:
    print("No labels to save.")
