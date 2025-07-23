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


#now to loop thru each filter (i, g, r)
for filt in ["i", "g", "r"]:
    df_filt = df[df["filter"] == filt]
    #send the output to..
    output_dir = os.path.join(output_base, f"training_dir_{filt}", "cv")
    os.makedirs(output_dir, exist_ok=True)

    #now loop per star (via id column)
    for star_id, group in df_filt.groupby("id"):
        out_path = os.path.join(output_dir, f"{star_id}.dat")
        group[["time", "mag", "mag_err"]].to_csv(out_path, sep=" ", index=False, header=False)
