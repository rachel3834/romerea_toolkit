from MicroLIA import training_set, ensemble_model
import os
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

filter_used = "g"

#base path for training data folders
base_training_path = "/data01/aschweitzer/software/microlia_output"


training_data_path = os.path.join(base_training_path, f"training_data_{filter_used}")
model_dir_name = os.path.join(base_training_path, f"model_{filter_used}")
csv_path = os.path.join(base_training_path, f"MicroLIA_Training_Set_{filter_used}.csv")


#load training data via microlia, save to .csv file for conf maatrix
data_x, data_y = training_set.load_all(path=training_data_path)
rows = []

for i, (lc, label) in enumerate(zip(data_x, data_y)):
    # lc should be a Nx3 numpy array or list of [time, mag, mag_err]
    print(type(lc), np.shape(lc))  #what shape is the data??? (168, 1) vs (168, 3) error check
    
    arr = np.array(lc)
    print(f"Original shape: {arr.shape}, dtype: {arr.dtype}")
    
    #if it's 1D but each element is an array/list, stack them to be 2D!
    if arr.ndim == 1 and isinstance(arr[0], (list, np.ndarray)):
        try:
            arr = np.vstack(arr)
        except Exception as e:
            print(f"[{i}] Could not stack lightcurve: {e}")
            continue
    
    #sanity check on shape
    if arr.ndim != 2 or arr.shape[1] != 3:
        print(f"[{i}] Skipping due to invalid shape: {arr.shape}")
        continue

    df_lc = pd.DataFrame(lc, columns=["time", "mag", "mag_err"])
    df_lc["label"] = label
    rows.append(df_lc)



if not rows:
    print("[ERROR] No valid lightcurves to concatenate. Check data_x logic.")
    print(f"data_x length: {len(data_x)}")
    print(f"{type(data_x[0]) if len(data_x) > 0 else 'N/A'}")
    exit()



df = pd.concat(rows, ignore_index=True)
df.to_csv(csv_path, index=False)

print(f"saved training data CSV to: {csv_path}")



print(f"Loaded {len(data_x)} lightcurves for filter g")

#create and optimize the model
model = ensemble_model.Classifier(
    data_x,
    data_y,
    impute=True,
    optimize=True,
    opt_cv=3,
    boruta_trials=25,
    n_iter=25
)

#train the model
model.create()
model.save(dirname=model_dir_name)
print(f"Model saved to: {model_dir_name}")

#load model from save state
loaded_df = pd.read_csv(csv_path)
model_loaded = ensemble_model.Classifier(clf='xgb', impute=True, training_data=loaded_df)
model_loaded.load(path=model_dir_name)
print(f"[INFO] Model reloaded from disk.")

#check what's being loaded
bad_files = []

for root, dirs, files in os.walk(training_data_path):
    for file in files:
        if file.endswith(".dat"):
            path = os.path.join(root, file)
            try:
                _ = training_set.load_single(path)
            except Exception as e:
                print(f"[BAD FILE] {path} -- {e}")
                bad_files.append(path)

print(f"\nTotal broken files: {len(bad_files)}")


#then clean them
def clean_broken_dat_files(training_data_path):
    removed = []
    for root, dirs, files in os.walk(training_data_path):
        for file in files:
            if file.endswith(".dat"):
                path = os.path.join(root, file)
                try:
                    _ = training_set.load_single(path)
                except:
                    print(f"Removing broken file: {path}")
                    os.remove(path)
                    removed.append(path)
    print(f"\nRemoved {len(removed)} broken .dat files.")

clean_broken_dat_files(training_data_path)


y_pred = model_loaded.predict(loaded_df)
y_true = loaded_df["label"]



#make conf matrix
labels = sorted(set(y_true))
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)


#plotting
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - Filter g")
plt.tight_layout()
final_path = os.path.join(base_training_path, "confusion_matrices")
os.makedirs(final_path, exist_ok=True)
plt.savefig(os.path.join(final_path, f"confusion_matrix_{filter_used}.png"))
plt.show()

print(f"saved this model to {final_path}!")