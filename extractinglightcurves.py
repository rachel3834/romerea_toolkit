import os
import shutil
import pandas as pd
import subprocess
import h5py
import traceback
import bz2


#paths
data_dir = "path"
photo_copy_dir = "another path"
lightcurve_out_dir = "final path"


os.makedirs(photo_copy_dir, exist_ok=True)
os.makedirs(lightcurve_out_dir, exist_ok=True)

def copy_and_unzip(field_num, quad_num, src_dir=data_dir, dest_dir=photo_copy_dir):
    filename = f"ROME-FIELD-{field_num}_quad{quad_num}_photometry.hdf5.bz2"
    src_path = os.path.join(src_dir, filename)
    dest_bz2_path = os.path.join(dest_dir, filename)
    dest_hdf5_path = dest_bz2_path[:-4] #removes .bz2

    if not os.path.exists(src_path):
        print("Source not found")
        return None
    shutil.copy(src_path, dest_bz2_path)
    print(f"Copied to {dest_bz2_path}")

    with bz2.BZ2File(dest_bz2_path, 'rb') as bz2_file, open(dest_hdf5_path, 'wb') as hdf5_file:
        shutil.copyfileobj(bz2_file, hdf5_file)

    print(f"Unzipped to {dest_hdf5_path}")
    return dest_hdf5_path



print("Now getting photometry...")


#processing
for fname in os.listdir(data_dir):
    if not fname.lower().endswith(".txt") or "cv_matches" not in fname.lower():
        continue

    print(f"Processing match file: {fname}!")

    match_path = os.path.join(data_dir, fname)
    df = pd.read_csv(match_path, delim_whitespace=True)
    df.columns = df.columns.str.lower()

    try:
        field_str = fname.split("-")[2].split("_")[0]
        field_num = int(field_str)

    except Exception as e:
        print(f"Failed to parse field from {fname}: {e}")
        continue

    if field_num in []:
        print(f"Skipping already copied field {field_num}")
        continue

    #now grouping rows by quadrant
    grouped = df.groupby("quadrant")
    for quadrant, group in grouped:
        hdf5_path = copy_and_unzip(field_num, quadrant)
        if hdf5_path is None:
            print(f"Skipping because photometry is missing")

        try:
            with h5py.File(hdf5_path, "r") as hdf:
                breakpoint()
                if "photometry" in hdf:
                    dset = hdf["dataset_photometry"]

                else:
                    first_key = list(hdf.keys())[0] if hdf.keys() else None
                    if not first_key:
                        print(f"No datasets found in {hdf5_path}")
                        continue
                    dset = hdf[first_key]

                data = {key: dset[key][:] for key in dset.dtype.names}
                photo_df = pd.DataFrame(data)
                photo_df.columns = [c.lower() for c in photo_df.columns]

                qids = group["quadrant_id"].astype(str).tolist()
                matched_df = photo_df[photo_df["quadrant_id"].astype(str).isin(qids)]

                if matched_df.empty:
                    print(f"No matches found")
                    continue
                for _, row in group.iterrows():
                    obj_id = row["gaia_source_id"]
                    qid = str(row["quadrant_id"])

                    lc_rows = matched_df[
                        (matched_df["gaia_source_id"] == obj_id)
                        & (matched_df["quadrant_id"].astype(str) == qid)
                    ]

                    if lc_rows.empty:
                        print("No lightcurves.")
                        continue
                
                    out_path = os.path.join(lightcurve_out_dir, f"{obj_id}_lc.txt")
                    lc_rows.to_csv(out_path, sep="\t", index=False)
                    print(f"Wrote lightcurve for {obj_id} to {out_path}")

        except Exception as e:
            print(f"Error reading {hdf5_path}: {e}!")