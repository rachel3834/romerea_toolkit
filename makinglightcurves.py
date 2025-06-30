import sys
sys.path.append("/data01/aschweitzer/software")
import astropy.io
from astropy.io import fits
import numpy as np
import os
import pandas as pd
from romerea_toolkit.hd5_utils import read_star_from_hd5_file
import matplotlib.pyplot as plt
from romerea_toolkit import crossmatch

#making paths now
photo_dir = "/data01/aschweitzer/software/photo_copies"
output_dir = "/data01/aschweitzer/software/CV_Lightcurves/plots"
data_dir = "/data01/aschweitzer/data"

os.makedirs(output_dir, exist_ok=True)


#looping
for field_num in range(1, 21):
        print(f"Processing field {field_num}")

        #getting crossmatch file
        crossmatch_temp = "/data01/aschweitzer/data/ROME/ROME-FIELD-{:02d}/ROME-FIELD-{:02d}_field_crossmatch.fits"
        crossmatch_file = crossmatch_temp.format(field_num, field_num)
        match_file = os.path.join(data_dir, f"ROME-FIELD-{field_num:02d}_CV_matches.txt")
        if not os.path.exists(match_file):
                print(f"Crossmatch msising!")
                continue
        #now loading in cv matches and crossmatch info
        print(f"loading cv matches and crossmatch info!")

        match_df = pd.read_csv(match_file, delim_whitespace=True)


        #loading field crossmatch table
        xmatch = crossmatch.CrossMatchTable()
        xmatch.load(crossmatch_file, log=None)
        image_data = xmatch.images
        hjd = image_data["hjd"]
        filters = image_data["filter"]

        grouped = match_df.groupby("quadrant")
        for quadrant, group in grouped:
                phot_file = os.path.join(photo_dir, f"ROME-FIELD-{field_num:02d}_quad{quadrant}_photometry.hdf5")

        if not os.path.exists(phot_file):
                print(f"Missing photometry")
                continue
        for _, row in group.iterrows():
                gaia_id = row["gaia_source_id"]
                qid = int(row["quadrant_id"])

                try:
                        star_lc = read_star_from_hd5_file(phot_file, qid)

                except Exception as e:
                        print(f"Failed reading quadrant {qid} from {phot_file}: {e}")
                        continue
                
                norm_mag = star_lc[:, 7]
                norm_err = star_lc[:, 8]


                #plotting by filter!!!

                print("Now making and saving lightcurve...")

                final_path = os.pathjoin(output_dir, f"Gaia_{gaia_id}_field{field_num}_quad{quadrant}_qid{qid}")
                df = pd.DataFrame({
                        "HJD": hjd,
                        "Normalize_Mag": norm_mag,
                        "Normalized_Mag_Err": norm_err,
                        "Filter": filters
                })

                df.to_csv(final_path, index=False, sep=' ', float_format='%.6f')
                print(f"All done with Gaia_{gaia_id}! Moving on...")