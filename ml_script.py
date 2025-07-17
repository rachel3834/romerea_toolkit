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

#loading input csv
#cSV has: id,fits_path,label (label is optional)
input_csv = "/data01/aschweitzer/software/CV_Lightcurves/ml_lcs/ml_table.csv"  
input_df = pd.read_csv(input_csv)


#ensure 'id' is a column (not an index) since error was encountered
if 'id' not in input_df.columns:
    input_df.reset_index(inplace=True)
    if 'id' not in input_df.columns:
        raise KeyError("'id' column not found in CSV even after reset_index.")


#store final data 
lightcurve_rows = []
labels = []

for _, row in tqdm(input_df.iterrows(), total=len(input_df)):
    star_id = row["id"]
    fits_path = row["fits_path"]
    label = row.get("label", None)

    local_filename = os.path.join(OUTPUT_DIR, os.path.basename(fits_path))

    #skip if already downloaded
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

    #now open FITS file
    try:
        with fits.open(local_filename) as hdul:
            for filt, name in zip(['i', 'g', 'r'], ['table_i', 'table_g', 'table_r']):
                if name not in hdul:
                    continue

                try:
                    data = Table(hdul[name].data).to_pandas()
                    if not {'Heliocentric Julian Date', 'Normalized mag', 'Normalized mag (error)'}.issubset(data.columns):
                        continue
                    
                    df = pd.DataFrame({
                        'id': star_id,
                        'time': data['Heliocentric Julian Date'],
                        'mag': data['Normalized mag'],
                        'mag_err': data['Normalized mag (error)'],
                        'filter': filt
                    })
                    lightcurve_rows.append(df)
                    
                except Exception as e_inner:
                    print(f"Error reading filter {filt} in {local_filename}: {e_inner}")
                    continue

            if label is not None:
                labels.append({"id": star_id, "label": label})

    except Exception as e:
        print(f"Error processing {local_filename}: {e}")

#here, save combined lightcurve CSV for microlia
if lightcurve_rows:
    lc_df = pd.concat(lightcurve_rows)
    lc_df.to_csv(LC_CSV, index=False)
    print(f"Saved lightcurves to {LC_CSV}")
else:
    print("No lightcurve data extracted.")

#now save label CSV for microlia
if labels:
    label_df = pd.DataFrame(labels)
    label_df.to_csv(LABEL_CSV, index=False)
    print(f"Saved labels to {LABEL_CSV}")
