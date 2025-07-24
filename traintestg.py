from MicroLIA import training_set, ensemble_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import shutil
import tempfile

filter_used = "g"
base_training_path = "/data01/aschweitzer/software/microlia_output"
training_data_path = os.path.join(base_training_path, f"training_data_{filter_used}")
final_model_dir = os.path.join(base_training_path, f"model_{filter_used}")

#load data via microlia
x, y = training_set.load_all(training_data_path)

#train model on training_data_path
model = ensemble_model.Classifier(
    x,
    y,
    impute=True,
    optimize=True,
    opt_cv=3,
    boruta_trials=25,
    n_iter=25
)
model.create()

#saving to a temp path
with tempfile.TemporaryDirectory(dir=base_training_path) as temp_dir:
    model.save(temp_dir)

    #remove previous model if it exists
    if os.path.exists(final_model_dir):
        shutil.rmtree(final_model_dir)

    #move new model from temp to final location
    shutil.move(temp_dir, final_model_dir)

print(f"Model saved to {final_model_dir}")


#evaluate this model
model_loaded = ensemble_model.Classifier(x, y, clf="xgb", impute=True)
model_loaded.load(final_model_dir)
y_pred = model_loaded.predict(x)

#make conf matrix
labels = sorted(set(y))
cm = confusion_matrix(y, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - Filter {filter_used}")
plt.tight_layout()

#SAVE confusion matrix
conf_matrix_dir = os.path.join(base_training_path, "confusion_matrices")
os.makedirs(conf_matrix_dir, exist_ok=True)
cm_path = os.path.join(conf_matrix_dir, f"confusion_matrix_{filter_used}.png")
plt.savefig(cm_path)
plt.show()

print(f"Confusion matrix saved to: {cm_path}")
