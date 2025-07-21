from MicroLIA import training_set, ensemble_model
from numpy.typing import ArrayLike

#training data folder
training_data_path = "/data01/aschweitzer/software/microlia_output/training_data"

#--------CHECKKKK!!!!---------#
import os
import pandas as pd

root = "/data01/aschweitzer/software/microlia_output/training_data"
X = []
Y = []

for label in ['CONST', 'rrlyr', 'lpv', 'ml', 'CV']:
    subdir = os.path.join(root, label)
    for fname in os.listdir(subdir):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(subdir, fname)
        try:
            df = pd.read_csv(path)
            if set(['time', 'mag', 'mag_err', 'filter']).issubset(df.columns) and not df.empty:
                X.append(df)
                Y.append(label)
                print(f"Loaded {fname}")
            else:
                print(f"Skipped {fname}: missing columns or empty")
        except Exception as e:
            print(f"Failed to load {fname}: {e}")

print(f"\nTotal loaded: {len(X)} lightcurves")
#-------------!!!!!------------#

data_x, data_y = training_set.load_all(path=training_data_path)


#now create the ensemble model with feature extraction and optimization
model = ensemble_model.Classifier(
    data_x,
    data_y,
    impute=True,          #handle missing values
    optimize=True,        #perform hyperparameter opt.
    opt_cv=3,             #num. of CV folds during opt.
    boruta_trials=25,     #boruta feature selection trials
    n_iter=25             #number of iter. for ensemble opt.
)

#train the model
model.create()

#save the model to disk so we can reuse it later w/o retraining
import pickle
with open("/data01/aschweitzer/software/microlia_output/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training complete and model saved.")
