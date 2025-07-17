import os
import pandas as pd
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

#base URL prefix for fits files (adjust if needed)
base_url = "https://exoplanetarchive.ipac.caltech.edu/data/ROME/data_reduction"

#reading input CSV
df = pd.read_csv(input_csv)

#storage for microlia lightcurve rows and IDs
all_lc_rows = []
all_ids = set()

for i, row in tqdm(df.iterrows(), total=len(df)):
    fits_path = row.get("fits", "")
    if not isinstance(fits_path, str) or not fits_path.endswith(".fits"):
        print(f"Skipping row {i}: invalid FITS path")
        continue

    #normalize fits_path, remove leading "./"
    if fits_path.startswith("./"):
        fits_path = fits_path[2:]

    full_url = f"{base_url}/{fits_path}"
    filename = os.path.basename(fits_path)
    local_path = os.path.join(output_dir, filename)

    #download if not already
    if not os.path.exists(local_path):
        try:
            temp_path = download_file(full_url, cache=True, show_progress=False, timeout=30)
            os.rename(temp_path, local_path)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {full_url}: {e}")
            continue

    #open FITS and extract data from each filter extension
    try:
        with fits.open(local_path) as hdul:
            filters = ["g", "r", "i"]
            ext_names = ["table_g", "table_r", "table_i"]
            obj_id = os.path.splitext(filename)[0]

            for filter_name, ext_name in zip(filters, ext_names):
                if ext_name not in hdul:
                    print(f"Warning: {ext_name} missing in {filename}, skipping filter {filter_name}")
                    continue

                data = hdul[ext_name].data
                required_cols = ["HJD", "NORMALIZED_MAG", "NORMALIZED_MAG_ERR"]
                if not all(col in data.columns.names for col in required_cols):
                    print(f"Missing required columns in {ext_name} for {filename}, so now skipping filter {filter_name}")
                    continue

                hjd = data["HJD"]
                mag = data["NORMALIZED_MAG"]
                mag_err = data["NORMALIZED_MAG_ERR"]

                for t, m, e in zip(hjd, mag, mag_err):
                    all_lc_rows.append({
                        "id": obj_id,
                        "time": float(t),
                        "mag": float(m),
                        "mag_err": float(e),
                        "filter": filter_name
                    })
            all_ids.add(obj_id)
    except Exception as e:
        print(f"Error processing {local_path}: {e}")
        continue

# safve the combined lightcurve CSV in Microlia format
lightcurve_df = pd.DataFrame(all_lc_rows)
lightcurve_df = lightcurve_df[["id", "time", "mag", "mag_err", "filter"]]
lightcurve_csv_path = os.path.join(output_dir, f"{class_name}_microlia_lightcurves.csv")
lightcurve_df.to_csv(lightcurve_csv_path, index=False)

#save labels CSV
label_df = pd.DataFrame({
    "id": sorted(all_ids),
    "label": [class_name] * len(all_ids)
})
label_csv_path = os.path.join(output_dir, f"{class_name}_microlia_labels.csv")
label_df.to_csv(label_csv_path, index=False)

print(f"\nSaved {len(all_ids)} stars' lightcurves to {lightcurve_csv_path}")
print(f"Saved labels to {label_csv_path}")
