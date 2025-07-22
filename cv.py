import os
import pandas as pd
from astropy.io import fits
from romerea_toolkit import crossmatch

#directories
match_dir = "/data01/aschweitzer/data/"
crossmatch_dir = "/data01/aschweitzer/data/ROME/"
output_root = "/data01/aschweitzer/software/microlia_output/"

#rome filter name --> microlia dir name
filter_map = {"ip": "i", "gp": "g", "rp": "r"}

#loop over 1-20 fields
for field_num in range(1, 25):
    field_str = f"ROME-FIELD-{field_num:02d}"
    match_file = os.path.join(match_dir, f"{field_str}_CV_matches.txt")
    crossmatch_file = os.path.join(crossmatch_dir, f"{field_str}_field_crossmatch.fits")

    if not os.path.exists(match_file) or not os.path.exists(crossmatch_file):
        print(f"Missing match file data for field {field_num}")
        continue

    print(f"Processing {field_str}")

    #load match file, field_id and label columns
    match_df = pd.read_csv(match_file, delim_whitespace=True)

    #load image data from crossmatch
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(crossmatch_file, log=None)
    image_data = xmatch.images

    #process each matched star
    for _, row in match_df.iterrows():
        fid = row["field_id"]

        star_obs = image_data[image_data["field_id"] == fid]
        if len(star_obs) == 0:
            continue

        for rome_filt, microlia_filt in filter_map.items():
            obs = star_obs[star_obs["filter"].astype(str) == rome_filt]
            if len(obs) == 0:
                continue

            df = pd.DataFrame({
                "time": obs["hjd"],
                "mag": obs["mag"],
                "mag_err": obs["mag_err"]
            })

            #output file path as space-sep .dat, no header
            out_dir = os.path.join(output_root, f"training_data_{microlia_filt}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{fid}.dat")
            print(f"successfully saved {field_str} as {out_path}! moving on....")

            df.to_csv(out_path, sep=" ", index=False, header=False, float_format="%.6f")

print(f"finished processing all matched cv stars for microlia! please check the output_dir for results!")
