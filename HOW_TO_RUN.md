# How to Run the Laptop Recommender System

## Quick Start

### Option 1: Run with Built-in Examples (Easiest)

```bash
python laptop_recommender.py
```

This will:
- Load the dataset
- Initialize the recommender system
- Show 3 example recommendations

### Option 2: Run the Interactive Examples

```bash
python project.ipy
```

This shows more detailed examples with explanations.

---

## Using in Your Own Code

### Step 1: Import the recommender

```python
from laptop_recommender import recommend_laptops
```

### Step 2: Get recommendations

```python
# Example 1: Find similar laptops
recommendations = recommend_laptops(
    reference_laptop_name="HP Victus",
    n_recommendations=5
)

# Example 2: Filter by preferences
recommendations = recommend_laptops(
    max_price=60000,
    use_case='gaming',
    min_rating=4.3,
    n_recommendations=5
)

# Example 3: Hybrid approach
recommendations = recommend_laptops(
    reference_laptop_name="MacBook",
    max_price=100000,
    min_ram=16,
    n_recommendations=5
)
```

### Step 3: Display results

```python
if isinstance(recommendations, str):
    print(recommendations)  # Error message
else:
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   {rec['brand']} | {rec['price']} | Rating: {rec['rating']}")
        print(f"   {rec['ram']} | {rec['storage']}")
        print(f"   {rec['processor']} | {rec['graphics']}")
```

---

## Available Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `reference_laptop_name` | str | Find laptops similar to this | `"HP Victus"` |
| `max_price` | float | Maximum price in rupees | `60000` |
| `min_rating` | float | Minimum rating (0-5) | `4.3` |
| `brand` | str | Filter by brand | `"HP"` |
| `min_ram` | int | Minimum RAM in GB | `16` |
| `min_storage` | int | Minimum storage in GB | `512` |
| `processor_brand` | str | Filter by processor | `"Intel"`, `"AMD"`, `"Apple"` |
| `graphics_brand` | str | Filter by graphics | `"NVIDIA"`, `"AMD"` |
| `use_case` | str | Use case filter | `"gaming"`, `"business"`, `"budget"` |
| `n_recommendations` | int | Number of results | `5` |

---

## Example Use Cases

### Gaming Laptops
```python
recommend_laptops(
    use_case='gaming',
    max_price=80000,
    min_rating=4.3
)
```

### Business Laptops
```python
recommend_laptops(
    use_case='business',
    min_ram=16,
    min_rating=4.4
)
```

### Budget Laptops
```python
recommend_laptops(
    use_case='budget',
    min_rating=4.5
)
```

### Specific Brand
```python
recommend_laptops(
    brand="Apple",
    max_price=150000
)
```

### Similar to a Laptop You Like
```python
recommend_laptops(
    reference_laptop_name="HP Victus 15",
    max_price=70000,
    n_recommendations=10
)
```

---

## Requirements

Make sure you have these installed:

```bash
pip install pandas numpy scikit-learn
```

---

## Troubleshooting

### Error: "No laptops found matching your criteria"
- Try relaxing your filters (lower min_rating, higher max_price)
- Remove some constraints

### Error: "Could not find laptop matching..."
- The reference laptop name might not be exact
- Try a partial name (e.g., "Victus" instead of full name)

### Import Error
- Make sure `laptop_recommender.py` is in the same directory
- Make sure `laptop_cleaned2.csv` is in the same directory

---

## Interactive Python Session

You can also use it interactively:

```python
# Start Python
python

# Import
from laptop_recommender import recommend_laptops

# Get recommendations
recs = recommend_laptops(max_price=50000, use_case='gaming')

# Explore results
for rec in recs:
    print(rec['name'], rec['price'])
```

