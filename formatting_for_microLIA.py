import os
import pandas as pd

#paths to combined csv files
combined_lc_csv = "/data01/aschweitzer/software/microlia_output/combined/microlia_combined_lightcurves.csv"
combined_label_csv = "/data01/aschweitzer/software/microlia_output/combined/microlia_combined_labels.csv"

#output base directory for MicroLIA data
output_base = "/data01/aschweitzer/software/microlia_output/training_data"
os.makedirs(output_base, exist_ok=True)

#load data
lc_df = pd.read_csv(combined_lc_csv)
labels_df = pd.read_csv(combined_label_csv)

#mMap star id to label for quick lookup
id_to_label = dict(zip(labels_df["id"], labels_df["label"]))

#group lightcurves by star id
grouped = lc_df.groupby("id")

for star_id, group in grouped:
    if star_id not in id_to_label:
        print(f"Warning: No label for star {star_id}, skipping.")
        continue
    
    label = id_to_label[star_id]
    star_dir = os.path.join(output_base, label)
    os.makedirs(star_dir, exist_ok=True)
    
    #now making filename
    filename = f"star_{star_id}.csv"
    filepath = os.path.join(star_dir, filename)
    


    group_clean = group[["time", "mag", "mag_err", "filter"]].copy()

    #force numeric conversion else NaN
    group_clean["time"] = pd.to_numeric(group_clean["time"], errors='coerce')
    group_clean["mag"] = pd.to_numeric(group_clean["mag"], errors='coerce')
    group_clean["mag_err"] = pd.to_numeric(group_clean["mag_err"], errors='coerce')

    #drop any rows where numeric conversion failed
    group_clean = group_clean.dropna(subset=["time", "mag", "mag_err"])

    group_clean = group_clean[group_clean["mag"] >= 0]

    # Replace gp/rp/ip → g/r/i
    group_clean["filter"] = group_clean["filter"].replace({"gp": "g", "rp": "r", "ip": "i"})

    #ensure filter column is labeled right
    valid_filters = {"g", "r", "i"}
    group_clean["filter"] = group_clean["filter"].astype(str).str.strip()

    #filter to only valid filters
    group_clean = group_clean[group_clean["filter"].isin(valid_filters)]

    if group_clean.empty:
        print(f"Skipping star {star_id}: no valid rows after cleaning.")
        continue


    group_clean.to_csv(filepath, index=False, header=False)
   

print("Conversion complete :)!")
