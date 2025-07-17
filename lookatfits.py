import requests
from astropy.io import fits
from io import BytesIO

url = "https://exoplanetarchive.ipac.caltech.edu/workspace/TMP_Z62BKO_8520/ROME/tab1/data/data_reduction/ROME-FIELD-01/FieldID021000_022000/ROME-FIELD-01_star_021691.fits"

# Download the FITS file in-memory
response = requests.get(url)
response.raise_for_status()

# Open FITS from bytes buffer
hdul = fits.open(BytesIO(response.content))

print("FITS HDU list:")
hdul.info()

# Print columns of each HDU that is a table
for hdu in hdul:
    if isinstance(hdu, fits.BinTableHDU) or isinstance(hdu, fits.TableHDU):
        print(f"\nHDU name: {hdu.name}")
        print(hdu.columns)

hdul.close()
