# Algorithms Used in the Laptop Recommender System

## Overview

The recommender system uses a **hybrid approach** combining multiple algorithms and techniques:

1. **Content-Based Filtering** using **Cosine Similarity**
2. **Knowledge-Based Filtering** using **Constraint-Based Filtering**
3. **Hybrid Scoring Algorithm** combining similarity and ratings

---

## 1. Content-Based Filtering Algorithm

### Core Algorithm: **Cosine Similarity**

**Formula:**
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where:
- `A` and `B` are feature vectors representing laptops
- `A · B` is the dot product
- `||A||` and `||B||` are the magnitudes (L2 norms)

### How It Works:

1. **Feature Engineering:**
   - Each laptop is represented as a feature vector with ~22 dimensions
   - Numerical features (Price, RAM, Storage, etc.) are normalized using **StandardScaler** (Z-score normalization)
   - Categorical features (Brand, Processor, etc.) are encoded using **LabelEncoder** and then normalized
   - Binary features (Touch screen, Integrated graphics) are converted to 0/1

2. **Similarity Calculation:**
   - Uses `sklearn.metrics.pairwise.cosine_similarity()`
   - Measures the cosine of the angle between two feature vectors
   - Returns values between -1 and 1 (typically 0 to 1 for normalized features)
   - Higher values = more similar laptops

3. **Ranking:**
   - Sorts laptops by similarity score (descending)
   - Returns top N most similar laptops

**Why Cosine Similarity?**
- Works well with high-dimensional feature vectors
- Normalized, so scale differences don't dominate
- Measures direction (similarity) rather than magnitude
- Fast to compute

---

## 2. Knowledge-Based Filtering Algorithm

### Algorithm: **Constraint-Based Filtering**

This is a rule-based filtering system that applies user-defined constraints.

### How It Works:

1. **Sequential Filtering:**
   - Applies filters one by one:
     - Price constraint: `Price <= max_price`
     - Rating constraint: `Rating >= min_rating`
     - Brand filter: String matching
     - Hardware filters: `RAM >= min_ram`, `Storage >= min_storage`
     - Use case filters: Domain-specific rules

2. **Use Case Rules:**
   - **Gaming**: Must have NVIDIA graphics (≥4GB) OR AMD RX dedicated graphics (≥4GB)
   - **Business**: Must have ≥8GB RAM
   - **Budget**: Price ≤ ₹50,000

3. **Result:**
   - Returns indices of laptops that pass all filters
   - Acts as a pre-filter before similarity calculation

**Why Constraint-Based?**
- Directly addresses user requirements
- Transparent and explainable
- No training needed
- Works well for cold start scenarios

---

## 3. Hybrid Scoring Algorithm

### Algorithm: **Weighted Linear Combination**

This combines content-based similarity with product ratings.

### When Reference Laptop is Provided:

**Formula:**
```
final_score = 0.7 × cosine_similarity + 0.3 × normalized_rating
```

**Steps:**
1. Calculate cosine similarity between reference laptop and filtered candidates
2. Normalize ratings to 0-1 scale: `(rating - min) / (max - min)`
3. Combine with weighted sum (70% similarity, 30% rating)
4. Rank by final score

**Why 70/30 Split?**
- Prioritizes feature similarity (what user wants)
- Still considers quality (ratings)
- Can be adjusted based on preference

### When No Reference Laptop:

**Formula:**
```
final_score = 0.7 × normalized_rating + 0.3 × price_score
```

Where:
- `price_score = 1 - (price / max_price)` (normalized, higher = cheaper)
- Favors high ratings and lower prices

---

## 4. Feature Engineering Techniques

### A. StandardScaler (Z-score Normalization)

**Formula:**
```
normalized_value = (x - μ) / σ
```

Where:
- `μ` = mean of the feature
- `σ` = standard deviation

**Purpose:** Normalizes numerical features to have mean=0, std=1

### B. LabelEncoder

**Purpose:** Converts categorical strings to numerical codes

**Example:**
- "HP" → 0
- "Dell" → 1
- "Lenovo" → 2

Then normalized: `(encoded - mean) / std`

### C. Feature Vector Construction

**Process:**
1. Extract numerical features → normalize each
2. Extract categorical features → encode → normalize
3. Extract binary features → convert to 0/1
4. Stack all features horizontally → create feature matrix

**Result:** Each laptop = 1 row vector of ~22 normalized features

---

## Algorithm Complexity

### Time Complexity:

- **Feature Engineering:** O(n × m) where n = laptops, m = features
- **Cosine Similarity:** O(n × d) where d = feature dimensions
- **Filtering:** O(n) for each filter
- **Overall:** O(n × m) - linear in dataset size

### Space Complexity:

- **Feature Matrix:** O(n × d) where d ≈ 22
- **Similarity Matrix:** O(n²) if storing all pairs (we don't - compute on demand)
- **Overall:** O(n × d) - efficient for large datasets

---

## Comparison with Other Algorithms

### Why NOT Collaborative Filtering?
- **Requires:** User-item interaction matrix
- **We have:** Only product ratings (no user-specific data)
- **Result:** Not feasible without user history

### Why NOT Matrix Factorization?
- **Requires:** User-item interactions
- **We have:** Only product features and ratings
- **Result:** Content-based is more appropriate

### Why NOT Deep Learning?
- **Requires:** Large dataset, training time, complexity
- **We have:** 1020 laptops, need fast recommendations
- **Result:** Cosine similarity is simpler, faster, and sufficient

---

## Summary

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| **Similarity** | Cosine Similarity | Find laptops with similar features |
| **Filtering** | Constraint-Based | Apply user preferences |
| **Scoring** | Weighted Linear Combination | Rank recommendations |
| **Normalization** | StandardScaler (Z-score) | Scale numerical features |
| **Encoding** | LabelEncoder | Convert categorical to numerical |

**Overall Approach:** Hybrid Content-Based + Knowledge-Based Recommender System

---

## References

- Cosine Similarity: Standard vector similarity measure
- Content-Based Filtering: Recommender Systems Handbook (Ricci et al.)
- Knowledge-Based Systems: Constraint-based recommendation

