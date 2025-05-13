import pickle

# Load pickled model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Extract and print parameters
print("coef:", model.coef_[0].tolist())
print("intercept:", model.intercept_[0])
print("scaler_mean:", scaler.mean_.tolist())
print("scaler_scale:", scaler.scale_.tolist())