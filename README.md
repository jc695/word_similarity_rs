# 🔡 Word Similarity Engine (Rust + Python)

This project provides a high-performance word similarity engine written in Rust, with a Python wrapper for easy integration. The model uses a logistic regression ensemble of four different string similarity algorithms.

---

## 📂 Project Structure

```
.
├── src/         # Core Rust module with similarity logic
├── python/      # Python bindings using PyO3
├── training/    # Training scripts and guide for generating model weights
```

---

## 🧠 Model Overview

The similarity score is based on a logistic regression model trained on four feature functions:

- **Jaccard Similarity** (n-grams)
- **Dice Coefficient** (n-grams)
- **LCS Similarity** (Longest Common Subsequence)
- **Possessive Similarity** (e.g., `Tim` vs `Tim's`)

The logistic model is trained in Python and hardcoded into `src/lib.rs` for maximum runtime performance.

---

## 🚀 Usage

### Build Rust Library

```bash
cargo build --release
```

### Use in Python

Install the Python wrapper:

```bash
cd python
uv pip install .
```

Example usage:

```python
from word_similarity_rs import predict_similarity

print(predict_similarity("Apple", "Apple's"))  # Output: ~0.84
```

---

## 🔁 Training the Model

To update the model or re-train with new data:

1. Go to the `training/` directory
2. Run the training script
3. Extract parameters and paste them into `src/lib.rs`

Follow the full guide here:

📄 [`training/TRAINING_GUIDE.md`](training/TRAINING_GUIDE.md)

---

## 🧪 Testing

Once rebuilt and installed, run tests from the Python wrapper:

```bash
uv run python test.py
```

---

## 🛠 Built With

- 🦀 Rust + [PyO3](https://pyo3.rs)
- 🐍 Python 3.9+
- 📊 scikit-learn (for training)
