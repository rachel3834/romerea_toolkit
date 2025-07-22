import os
import pandas as pd
import requests
from astropy.io import fits
from tqdm import tqdm

#constants and paths
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/workspace/TMP_Z62BKO_8520/ROME/tab1/data/data_reduction"
OUTPUT_DIR = "/data01/aschweitzer/software/microlia_output/rrlyr"
TRAINING_BASE = "/data01/aschweitzer/software/microlia_output/training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LC_CSV = os.path.join(OUTPUT_DIR, "rrlyr_microlia_lightcurves.csv")
LABEL_CSV = os.path.join(OUTPUT_DIR, "rrlyr_microlia_labels.csv")

input_csv = "/data01/aschweitzer/software/CV_Lightcurves/rrlyrae_lcs/rrlyr_table.csv"
input_df = pd.read_csv(input_csv)

label = "rrlyr"


# Map filters to FITS HDU names
filter_hdu_map = {
    "g": "LIGHTCURVE_SDSS_G",
    "r": "LIGHTCURVE_SDSS_R",
    "i": "LIGHTCURVE_SDSS_I"
}

lightcurve_rows = []


for star_name, star_group in tqdm(input_df.groupby("name"), desc="Processing stars"):
    fits_path = star_group["lc_file_path"].iloc[0]
    local_filename = os.path.join(OUTPUT_DIR, os.path.basename(fits_path))

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

    #extract filter lightcurves from fits
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

                #format: time, mag, mag_err
                df = pd.DataFrame({
                    "time": hjd,
                    "mag": norm_mag,
                    "mag_err": norm_mag_err
                })

                # Save to training_data_{filter}/{label}/...
                per_filter_dir = os.path.join(f"{TRAINING_BASE}_{filt}", label)
                os.makedirs(per_filter_dir, exist_ok=True)

                filename = f"star_{star_name}.dat"
                filepath = os.path.join(per_filter_dir, filename)
                df.to_csv(filepath, index=False, header=False, float_format="%.6f", sep=" ")
                print(f"Saved: {filepath}")

                #also store for combined CSV just for analysis
                df["id"] = star_name
                df["filter"] = filt
                lightcurve_rows.append(df)

    except Exception as e:
        print(f"Error reading FITS {local_filename}: {e}")
        continue


# save combined CSV (for analysis not training)
if lightcurve_rows:
    lightcurve_df = pd.concat(lightcurve_rows, ignore_index=True)
    lightcurve_df.to_csv(LC_CSV, index=False)
    print(f"Saved lightcurve CSV to {LC_CSV}")
