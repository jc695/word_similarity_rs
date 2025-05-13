use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::cmp::max;
use std::collections::HashSet;

// Model parameters from extract_parameters.py
const COEF: [f64; 4] = [0.6806876906244687, 0.6377520833850591, 0.6617739811501231, 0.6617739811501231];
const INTERCEPT: f64 = -0.7960397626745217;
const SCALER_MEAN: [f64; 4] = [0.31212042788129746, 0.39237943955685894, 0.4572420634920635, 0.4572420634920635];
const SCALER_SCALE: [f64; 4] = [0.3074975786341642, 0.3596572915443685, 0.2910227624475045, 0.2910227624475045];

// Jaccard similarity
fn jaccard_similarity(str1: &str, str2: &str, n_gram: usize) -> f64 {
    let trigrams1: HashSet<String> = str1
        .chars()
        .collect::<Vec<char>>()
        .windows(n_gram)
        .map(|w| w.iter().collect::<String>())
        .collect();
    let trigrams2: HashSet<String> = str2
        .chars()
        .collect::<Vec<char>>()
        .windows(n_gram)
        .map(|w| w.iter().collect::<String>())
        .collect();
    let intersection = trigrams1.intersection(&trigrams2).count() as f64;
    let union = trigrams1.union(&trigrams2).count() as f64;
    if union == 0.0 { 0.0 } else { intersection / union }
}

// Dice coefficient
fn dice_coefficient(str1: &str, str2: &str, n_gram: usize) -> f64 {
    let trigrams1: HashSet<String> = str1
        .chars()
        .collect::<Vec<char>>()
        .windows(n_gram)
        .map(|w| w.iter().collect::<String>())
        .collect();
    let trigrams2: HashSet<String> = str2
        .chars()
        .collect::<Vec<char>>()
        .windows(n_gram)
        .map(|w| w.iter().collect::<String>())
        .collect();
    let intersection = trigrams1.intersection(&trigrams2).count() as f64;
    let total = (trigrams1.len() + trigrams2.len()) as f64;
    if total == 0.0 { 0.0 } else { 2.0 * intersection / total }
}

// Longest Common Subsequence similarity
fn lcs_similarity(str1: &str, str2: &str) -> f64 {
    let s1: Vec<char> = str1.chars().collect();
    let s2: Vec<char> = str2.chars().collect();
    let m = s1.len();
    let n = s2.len();
    let mut dp = vec![vec![0; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if s1[i - 1] == s2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    let lcs_length = dp[m][n] as f64;
    let max_len = max(m, n) as f64;
    if max_len == 0.0 { 0.0 } else { lcs_length / max_len }
}

// Possessive similarity
fn possessive_similarity(str1: &str, str2: &str) -> f64 {
    let norm1 = str1.to_lowercase().replace("'s", "").replace("s", "");
    let norm2 = str2.to_lowercase().replace("'s", "").replace("s", "");
    if norm1 == norm2 {
        let len_diff = (str1.len() as i32 - str2.len() as i32).abs() as f64;
        let base_score = 1.0;
        let penalty = len_diff / max(str1.len(), str2.len()) as f64;
        return (base_score - penalty).max(0.0);
    }
    lcs_similarity(str1, str2)
}

// Sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Predict similarity
#[pyfunction]
fn predict_similarity(str1: &str, str2: &str) -> PyResult<f64> {
    let features = [
        jaccard_similarity(str1, str2, 3),
        dice_coefficient(str1, str2, 3),
        lcs_similarity(str1, str2),
        possessive_similarity(str1, str2),
    ];

    // Scale features
    let mut scaled_features = [0.0; 4];
    for i in 0..4 {
        scaled_features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }

    // Compute linear combination
    let mut linear_sum = INTERCEPT;
    for i in 0..4 {
        linear_sum += COEF[i] * scaled_features[i];
    }

    // Apply sigmoid
    Ok(sigmoid(linear_sum))
}

// Module definition
#[pymodule]
fn word_similarity_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(predict_similarity, m)?)?;
    Ok(())
}