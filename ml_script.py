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
input_csv = "/data01/aschweitzer/software/CV_Lightcurves/ml_lcs/lpv_table.csv"
input_df = pd.read_csv(input_csv)

#output containers
lightcurve_rows = []
labels = []

# Group by name and filter
grouped = input_df.groupby(["name", "filter", "fits_path"])

for (name, filt, fits_path), group in tqdm(grouped, total=len(grouped)):
    #now using the star 'name' as unique ID for microlia
    star_id = name

    #save label if exists
    label = group["label"].iloc[0] if "label" in group.columns else None
    if label is not None:
        labels.append({"id": star_id, "label": label})

    #download FITS if needed
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

    #get HJD from FITS
    try:
        with fits.open(local_filename) as hdul:
            table_name = f"table_{filt}"
            if table_name not in hdul:
                print(f"Missing {table_name} in {fits_path}")
                continue
            hjd = hdul[table_name].data["Heliocentric Julian Date"]
    except Exception as e:
        print(f"Error reading FITS {local_filename}: {e}")
        continue

    #now trim group to match HJD length
    group_sorted = group.sort_values("epoch").reset_index(drop=True)
    if len(hjd) != len(group_sorted):
        print(f"Length mismatch for {star_id} {filt}, skipping.")
        continue

    df = pd.DataFrame({
        "id": star_id,
        "time": hjd,
        "mag": group_sorted["Normalized mag"],
        "mag_err": group_sorted["Normalized mag (error)"],
        "filter": filt
    })
    lightcurve_rows.append(df)

#save lightcurve CSV
if lightcurve_rows:
    lc_df = pd.concat(lightcurve_rows, ignore_index=True)
    lc_df.to_csv(LC_CSV, index=False)
    print(f"Saved lightcurves to {LC_CSV}")
else:
    print("No lightcurve data extracted.")

#save label CSV
if labels:
    label_df = pd.DataFrame(labels)
    label_df.drop_duplicates(subset="id").to_csv(LABEL_CSV, index=False)
    print(f"Saved labels to {LABEL_CSV}")
