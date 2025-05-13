import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Similarity functions
def jaccard_similarity(str1, str2, n_gram=3):
    str1, str2 = str1.lower(), str2.lower()
    trigrams1 = {str1[i:i+n_gram] for i in range(len(str1)-n_gram+1)}
    trigrams2 = {str2[i:i+n_gram] for i in range(len(str2)-n_gram+1)}
    intersection = len(trigrams1 & trigrams2)
    union = len(trigrams1 | trigrams2)
    return intersection / union if union != 0 else 0.0

def dice_coefficient(str1, str2, n_gram=3):
    str1, str2 = str1.lower(), str2.lower()
    trigrams1 = {str1[i:i+n_gram] for i in range(len(str1)-n_gram+1)}
    trigrams2 = {str2[i:i+n_gram] for i in range(len(str2)-n_gram+1)}
    intersection = len(trigrams1 & trigrams2)
    total = len(trigrams1) + len(trigrams2)
    return (2 * intersection) / total if total != 0 else 0.0

def lcs_similarity(str1, str2):
    str1, str2 = str1.lower(), str2.lower()
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_length = dp[m][n]
    max_len = max(m, n)
    return lcs_length / max_len if max_len != 0 else 0.0

def possessive_similarity(str1, str2):
    norm1 = str1.lower().replace("'s", "").replace("s", "")
    norm2 = str2.lower().replace("'s", "").replace("s", "")
    if norm1 == norm2:
        len_diff = abs(len(str1) - len(str2))
        base_score = 1.0
        penalty = len_diff / max(len(str1), len(str2))
        return max(0.0, base_score - penalty)
    return lcs_similarity(str1, str2)

# Training data
training_data = [
    ("Tim", "Tim's", 1),
    ("Apple", "Apple's", 1),
    ("Gordon", "Gordon's", 1),
    ("Bestbuy", "Bestbuy's", 1),
    ("Lobster", "Lobsters", 1),
    ("Tim", "Gordon", 0),
    ("Apple", "Lobster", 0),
    ("Bestbuy", "Tim", 0),
    ("Gordon", "Apple", 0),
    ("Lobster", "Bestbuy", 0),
    ("Apple", "The Apple Bank", 0),
    ("Balcony Technology", "Pipe Technologies", 0)
]

# Compute features with progress bar
X = []
y = []
for w1, w2, label in tqdm(training_data, desc="Computing features"):
    features = [
        jaccard_similarity(w1, w2, 3),
        dice_coefficient(w1, w2, 3),
        lcs_similarity(w1, w2),
        possessive_similarity(w1, w2)
    ]
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved as model.pkl and scaler.pkl")