from MicroLIA import training_set, ensemble_model

#training data folder
training_data_path = "/data01/aschweitzer/software/microlia_output/training_data"

#get data and labels from folder structure
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
