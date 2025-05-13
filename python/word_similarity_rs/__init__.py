from .word_similarity_rs import predict_similarity as _predict_similarity

def predict_similarity(word1: str, word2: str) -> float:
    """
    Predict similarity score [0, 1] for a word pair.
    0 = unrelated, 1 = related (e.g., possessive/plural variants).
    """
    return _predict_similarity(word1, word2)
