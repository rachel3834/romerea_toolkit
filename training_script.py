from MicroLIA import training_set, ensemble_model
import os
import pickle

#base path for training data folders
base_training_path = "/data01/aschweitzer/software/microlia_output"

#list filters to train on
filters = ["i", "g", "r"]

#now loop over each filter
for filt in filters:
    training_data_path = os.path.join(base_training_path, f"training_data_{filt}")

    #load training data via microlia
    data_x, data_y = training_set.load_all(path=training_data_path, filters=[filt])

    print(f"Loaded {len(data_x)} lightcurves for filter {filt}")

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
    model_path = os.path.join(base_training_path, f"model_{filt}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Training complete and model for filter '{filt}' saved to {model_path}")
