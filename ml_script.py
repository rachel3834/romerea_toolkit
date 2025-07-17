import os
import pandas as pd
import requests
from astropy.io import fits
from tqdm import tqdm

#paths and consts
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/workspace/TMP_Z62BKO_8520/ROME/tab1/data/data_reduction"
OUTPUT_DIR = "/data01/aschweitzer/software/microlia_output/ml"
LC_CSV = "microlia_lightcurves.csv"
LABEL_CSV = "microlia_labels.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#load input
input_csv = "/data01/aschweitzer/software/CV_Lightcurves/ml_lcs/ml_table.csv"
input_df = pd.read_csv(input_csv)

#determine filters by csv table column names
mag_cols = [c for c in input_df.columns if c.startswith("norm_mag_")]
filters = [c.replace("norm_mag_", "") for c in mag_cols]

#output containers
lightcurve_rows = []
labels = []

#now iterate over unique star names and fits paths
for star_name, star_group in tqdm(input_df.groupby("name")):
    fits_path = star_group["fits_path"].iloc[0]  # assume unique per star

    #save label if exists
    label = star_group["label"].iloc[0] if "label" in star_group.columns else None
    if label is not None:
        labels.append({"id": star_name, "label": label})

    #download FITS if not already downloaded
    local_filename = os.path.join(OUTPUT_DIR, os.path.basename(fits_path))
    if not os.path.exists(local_filename):
        url = f"{BASE_URL}/{fits_path}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            continue

    #open FITS file once per star
    try:
        with fits.open(local_filename) as hdul:
            #for each filter, extract HJD and magnitudes
            for filt in filters:
                table_name = f"table_{filt}"
                if table_name not in hdul:
                    print(f"Missing {table_name} in {fits_path}")
                    continue

                #extract HJD
                hjd = hdul[table_name].data["Heliocentric Julian Date"]

                #extract mag and mag_err columns from input_df for this star and filter
                mag_col = f"norm_mag_{filt}"
                mag_err_col = f"norm_mag_err_{filt}"
                if mag_col not in star_group.columns or mag_err_col not in star_group.columns:
                    print(f"Missing mag or error columns for filter {filt}")
                    continue

                #now select and sort by epoch (if exists), else use index
                if "epoch" in star_group.columns:
                    star_filter_data = star_group.sort_values("epoch").reset_index(drop=True)
                else:
                    star_filter_data = star_group.reset_index(drop=True)

                #check that the length matches
                if len(hjd) != len(star_filter_data):
                    print(f"Length mismatch for star {star_name} filter {filt}, skipping.")
                    continue

                #now buuild output dataframe for this star and filter
                df = pd.DataFrame({
                    "id": star_name,
                    "time": hjd,
                    "mag": star_filter_data[mag_col],
                    "mag_err": star_filter_data[mag_err_col],
                    "filter": filt
                })

                lightcurve_rows.append(df)

    except Exception as e:
        print(f"Error reading FITS {local_filename}: {e}")
        continue

#save outputs for microlia format
if lightcurve_rows:
    lc_df = pd.concat(lightcurve_rows, ignore_index=True)
    lc_df.to_csv(LC_CSV, index=False)
    print(f"Saved lightcurves to {LC_CSV}")
else:
    print("No lightcurve data extracted.")

if labels:
    label_df = pd.DataFrame(labels).drop_duplicates(subset="id")
    label_df.to_csv(LABEL_CSV, index=False)
    print(f"Saved labels to {LABEL_CSV}")