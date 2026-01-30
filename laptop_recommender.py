"""
Main Laptop Recommender System
Imports and uses the modular components
"""

import pandas as pd
from preprocessing import DataPreprocessor
from models import LaptopRecommenderModel
from recommendations import RecommendationEngine
import warnings
warnings.filterwarnings('ignore')


# Initialize components
def initialize_recommender(data_path='laptop_cleaned2.csv'):
    """
    Initialize the complete recommender system
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file with laptop data
        
    Returns:
    --------
    RecommendationEngine
        Initialized recommendation engine
    """
    # Load data
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} laptops")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.clean_data(df)
    feature_matrix = preprocessor.create_feature_matrix(processed_df)
    print(f"Feature matrix created: {feature_matrix.shape}")
    
    # Initialize model
    model = LaptopRecommenderModel(processed_df, feature_matrix)
    
    # Initialize recommendation engine
    engine = RecommendationEngine(processed_df, feature_matrix, model)
    
    return engine, processed_df


# Initialize the recommender (available when imported)
recommender, processed_df = initialize_recommender()


def recommend_laptops(reference_laptop_name=None, max_price=None, min_rating=None,
                     brand=None, min_ram=None, min_storage=None,
                     processor_brand=None, graphics_brand=None,
                     use_case=None, n_recommendations=10):
    """
    Easy-to-use recommendation function
    
    Parameters:
    -----------
    reference_laptop_name : str, optional
        Name of a laptop to find similar ones to
    max_price : float, optional
        Maximum price in rupees
    min_rating : float, optional
        Minimum rating (0-5)
    brand : str, optional
        Filter by brand name
    min_ram : int, optional
        Minimum RAM in GB
    min_storage : int, optional
        Minimum storage in GB
    processor_brand : str, optional
        Filter by processor brand (Intel, AMD, Apple)
    graphics_brand : str, optional
        Filter by graphics brand (NVIDIA, AMD, Intel)
    use_case : str, optional
        'gaming', 'business', or 'budget'
    n_recommendations : int
        Number of recommendations to return
    
    Returns:
    --------
    list or str
        List of recommended laptops with details, or error message
    """
    laptop_idx = None
    
    # Find reference laptop if provided
    if reference_laptop_name:
        laptop_idx = recommender.find_laptop_by_name(reference_laptop_name)
        if laptop_idx is not None:
            print(f"Found reference laptop: {processed_df.iloc[laptop_idx]['Name']}\n")
        else:
            print(f"Warning: Could not find laptop matching '{reference_laptop_name}'. Proceeding without reference.\n")
    
    # Get recommendations
    recommendations = recommender.hybrid_recommend(
        laptop_idx=laptop_idx,
        max_price=max_price,
        min_rating=min_rating,
        brand=brand,
        min_ram=min_ram,
        min_storage=min_storage,
        processor_brand=processor_brand,
        graphics_brand=graphics_brand,
        use_case=use_case,
        n_recommendations=n_recommendations
    )
    
    return recommendations


if __name__ == "__main__":
    print("\n✅ Recommender system initialized successfully!")
    print(f"Feature matrix shape: {recommender.feature_matrix.shape}\n")
    
    # Example 1: Find similar laptops to a specific laptop
    print("="*80)
    print("Example 1: Find similar laptops to HP Victus")
    print("="*80)
    laptop_index = 0  # HP Victus 15-fb0157AX
    print(f"Reference Laptop: {processed_df.iloc[laptop_index]['Name']}")
    print(f"Price: ₹{processed_df.iloc[laptop_index]['Price']:,.0f}")
    print(f"Rating: {processed_df.iloc[laptop_index]['Rating']}")
    print("\nSimilar Laptops (Content-Based):")
    
    similar = recommender.model.get_similar_laptops(laptop_index, n_recommendations=5)
    for i, rec in enumerate(similar, 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   Brand: {rec['brand']} | Price: ₹{rec['price']:,.0f} | Rating: {rec['rating']}")
        print(f"   Similarity Score: {rec['similarity_score']:.3f}")
    
    # Example 2: Knowledge-based filtering
    print("\n" + "="*80)
    print("Example 2: Recommendations based on preferences")
    print("="*80)
    print("Criteria: Budget ≤ ₹60,000, Rating ≥ 4.5, Gaming use case")
    
    recommendations = recommend_laptops(
        max_price=60000,
        min_rating=4.5,
        use_case='gaming',
        n_recommendations=5
    )
    
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"   Brand: {rec['brand']} | Price: {rec['price']} | Rating: {rec['rating']}")
            print(f"   RAM: {rec['ram']} | Storage: {rec['storage']}")
            print(f"   Processor: {rec['processor']} | Graphics: {rec['graphics']}")
    
    # Example 3: Hybrid recommendation
    print("\n" + "="*80)
    print("Example 3: Hybrid Recommendations")
    print("="*80)
    print("Similar to reference laptop + Budget ≤ ₹70,000 + Rating ≥ 4.3")
    
    recommendations = recommend_laptops(
        reference_laptop_name="HP Victus",
        max_price=70000,
        min_rating=4.3,
        n_recommendations=5
    )
    
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"   Brand: {rec['brand']} | Price: {rec['price']} | Rating: {rec['rating']}")
            print(f"   RAM: {rec['ram']} | Storage: {rec['storage']}")
            print(f"   Processor: {rec['processor']} | Graphics: {rec['graphics']}")
            print(f"   Display: {rec['display']}")
