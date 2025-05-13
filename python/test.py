from word_similarity_rs import predict_similarity

pairs = [
    ("Tim", "Tim's"),
    ("Apple", "Apple's"),
    ("Gordon", "Gordon's"),
    ("Bestbuy", "Bestbuy's"),
    ("Lobster", "Lobsters"),
    ("Apple", "The Apple Bank"),
    ("Balcony Technology", "Pipe Technologies")
]

for w1, w2 in pairs:
    score = predict_similarity(w1, w2)
    print(f"Pair: '{w1}' vs. '{w2}' (lengths: {len(w1)}, {len(w2)}): Score = {score:.3f}")