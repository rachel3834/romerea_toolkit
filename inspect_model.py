import pickle
import os

base_path = "/data01/aschweitzer/software/microlia_output"
filters = ["i", "g", "r"]

#insepct for all three filters
for filt in filters:
    model_path = os.path.join(base_path, f"model_{filt}.pkl")
    
    if not os.path.isfile(model_path):
        print(f"[{filt}] Model file not found at: {model_path}")
        continue

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"[{filt}] Loaded model of type: {type(model)}")
    print(f"[{filt}] Model attributes: {list(model.__dict__.keys())}")

    #this is just a neat visual cutoff so i can easily see which filter loop ends where!!
    print("-" * 60)
