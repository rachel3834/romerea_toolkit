from MicroLIA import training_set, ensemble_model
import os
import pickle

#base path for training data folders
base_training_path = "/data01/aschweitzer/software/microlia_output"

#MUST SPECIFY FILTER i, g, r BELOW IN DESIGNATED AREAS
training_data_path = os.path.join(base_training_path, f"training_data_r")

#load training data via microlia
data_x, data_y = training_set.load_all(path=training_data_path)

print(f"Loaded {len(data_x)} lightcurves for filter r")

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

#save model
model_path = os.path.join(base_training_path, f"model_r.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Training complete and model saved to {model_path}")
