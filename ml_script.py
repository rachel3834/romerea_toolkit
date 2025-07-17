import os
import pandas as pd
import requests
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

#base download URL
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/workspace/TMP_Z62BKO_8520/ROME/tab1/data/data_reduction"
OUTPUT_DIR = "/data01/aschweitzer/software/microlia_output/ml"
LC_CSV = "microlia_lightcurves.csv"
LABEL_CSV = "microlia_labels.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#load input CSV: id,fits_path[,label]
input_csv = "/data01/aschweitzer/software/CV_Lightcurves/ml_lcs/ml_table.csv"  
input_df = pd.read_csv(input_csv)

lightcurve_rows = []
labels = []

for _, row in tqdm(input_df.iterrows(), total=len(input_df)):
    star_id = row["id"]
    fits_path = row["fits_path"]
    label = row.get("label", None)

    local_filename = os.path.join(OUTPUT_DIR, os.path.basename(fits_path))

    #download FITS if not already there
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

    #open FITS and extract normalized magnitudes per filter
    try:
        with fits.open(local_filename) as hdul:
            for filt, ext_name in zip(['i', 'g', 'r'], ['table_i', 'table_g', 'table_r']):
                if ext_name not in hdul:
                    continue

                data = Table(hdul[ext_name].data).to_pandas()

                required_cols = {'Heliocentric Julian Date', 'Normalized mag', 'Normalized mag (error)'}
                if not required_cols.issubset(data.columns):
                    print(f"Skipping {star_id}, {ext_name}: missing columns")
                    continue

                #now create dataframe and assign the ID manually from csv table
                df = pd.DataFrame({
                    'time': data['Heliocentric Julian Date'],
                    'mag': data['Normalized mag'],
                    'mag_err': data['Normalized mag (error)'],
                })
                df["id"] = star_id
                df["filter"] = filt

                lightcurve_rows.append(df)

            if label is not None:
                labels.append({"id": star_id, "label": label})

    except Exception as e:
        print(f"Error processing {local_filename}: {e}")

#save final lightcurve CSV for Microlia
if lightcurve_rows:
    lc_df = pd.concat(lightcurve_rows, ignore_index=True)
    lc_df = lc_df[["id", "time", "mag", "mag_err", "filter"]]
    lc_df.to_csv(LC_CSV, index=False)
    print(f"Saved lightcurves to {LC_CSV}")
else:
    print("No lightcurve data extracted.")

#save labels CSV
if labels:
    label_df = pd.DataFrame(labels)
    label_df.to_csv(LABEL_CSV, index=False)
    print(f"Saved labels to {LABEL_CSV}")
