import pandas as pd
import os

#this script works off of cv.py to turn the combined csv file cv_microlia_lightcurves.csv into .dat files for microlia

csv_path = "/data01/aschweitzer/software/microlia_output/cv/cv_microlia_lightcurves.csv"
output_base = "/data01/aschweitzer/software/microlia_output"


df = pd.read_csv(csv_path)

#reading .csv file now for header names
required_columns = {"id", "time", "mag", "mag_err", "filter"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing columns in input CSV. Required: {required_columns}")


#making sure mag and mag_err are float format
df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
df["mag_err"] = pd.to_numeric(df["mag_err"], errors="coerce")

#filter out anything <= 0.0 for mag and _err
df = df[(df["mag"] > 0.0) & (df["mag_err"] > 0.0)]

#now to loop thru each filter (i, g, r)
for filt in ["i", "g", "r"]:
    df_filt = df[df["filter"] == filt]
    #send the output to..
    output_dir = os.path.join(output_base, f"training_data_{filt}", "cv")
    os.makedirs(output_dir, exist_ok=True)

    #now loop per star (via id column)
    for star_id, group in df_filt.groupby("id"):
        out_path = os.path.join(output_dir, f"{star_id}.dat")
        group[["time", "mag", "mag_err"]].to_csv(out_path, sep=" ", index=False, header=False)
