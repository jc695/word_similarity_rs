# 🧠 Training Guide for `word_similarity_rs`

This guide walks you through training the logistic regression model and extracting parameters for use in the Rust-based similarity engine.

---

## 📁 Overview

The training process has two stages:

1. **Model Training**: Learn weights (coefficients and intercept) and scaler values from training data.
2. **Parameter Extraction**: Convert the trained model and scaler into Rust-compatible constants.

---

## 🛠️ Prerequisites

Ensure you have the following:

- Python 3.9+
- `uv` for dependency management (or `pip`)
- Rust toolchain (for later testing)
- Project structure intact (see repo layout)

Install dependencies:

```bash
cd training
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt  # Or just use `uv pip install scikit-learn joblib`
```

---

## 🔁 Step 1: Train the Model

This will:
- Load training data (defined inside `train_and_pickle.py`)
- Train a logistic regression model
- Fit a `StandardScaler`
- Save both to disk as `.pkl` files

```bash
cd training
python train_and_pickle.py
```

This will create:
- `model.pkl`
- `scaler.pkl`

---

## 📤 Step 2: Extract Model Parameters to Rust

Next, extract raw weights and scaling parameters for embedding directly into Rust:

```bash
python extract_parameters.py
```

This will print four constants:

- `COEF`
- `INTERCEPT`
- `SCALER_MEAN`
- `SCALER_SCALE`

### ✍️ Copy These Into `src/lib.rs`

Open `src/lib.rs` and replace the existing constants with the newly printed values.

Example:

```rust
const COEF: [f64; 4] = [ ... ];
const INTERCEPT: f64 = ...;
const SCALER_MEAN: [f64; 4] = [ ... ];
const SCALER_SCALE: [f64; 4] = [ ... ];
```

---

## ✅ Done! Rebuild and Test

After updating `lib.rs`, you can rebuild the Rust library and test:

```bash
cd ../
cargo build --release
cd python
uv run python test.py
```

---

## 📎 Notes

- You can customize `train_and_pickle.py` to include more training data or features.
- The extracted parameters are hardcoded in Rust to avoid runtime overhead.
