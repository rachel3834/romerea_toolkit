from MicroLIA import training_set, ensemble_model
import os
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#base path for training data folders
base_training_path = "/data01/aschweitzer/software/microlia_output"

#MUST SPECIFY FILTER i, g, r BELOW IN DESIGNATED AREAS
training_data_path = os.path.join(base_training_path, f"training_data_i")
model_path = os.path.join(base_training_path, f"model_i.pkl")


#load training data via microlia
data_x, data_y = training_set.load_all(path=training_data_path)

print(f"Loaded {len(data_x)} lightcurves for filter i")

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
model_path = os.path.join(base_training_path, f"model_i.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Training complete and model saved to {model_path}")



X_val, y_val = training_set.load_all(path=training_data_path)
print(f"Loaded {len(X_val)} validation lightcurves for filter i")

#predict using trained model
y_pred = model.predict(X_val)


#make conf matrix
labels = sorted(set(y_val))  # Ensure consistent label ordering
cm = confusion_matrix(y_val, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)


#plotting
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - Filter i")
plt.tight_layout()

final_path = os.path.join(base_training_path, "confusion_matrices")
os.makedirs(final_path, exist_ok=True)
plt.savefig(os.path.join(final_path, f"confusion_matrix_i.png"))
plt.show()